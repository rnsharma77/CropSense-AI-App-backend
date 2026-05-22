const path = require('path');
const crypto = require('crypto');
const fs = require('fs');
const dotenv = require('dotenv');
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const { MongoClient, ObjectId } = require('mongodb');
const { execFile, execSync } = require('child_process');

const ENV_PATHS = [
  path.resolve(__dirname, '.env.local'),
  path.resolve(__dirname, '.env'),
  // Backward-compatible fallback for older deployment layouts.
  path.resolve(__dirname, '../.env.local'),
  path.resolve(__dirname, '../.env'),
];

function loadEnvFiles({ override = false } = {}) {
  ENV_PATHS.forEach((envPath) => {
    dotenv.config({ path: envPath, override });
  });
}

function getGeminiApiKey() {
  if (!process.env.GEMINI_API_KEY) {
    loadEnvFiles({ override: false });
  }

  return process.env.GEMINI_API_KEY;
}

function extractGeminiReply(data) {
  return data?.candidates
    ?.flatMap((candidate) => candidate?.content?.parts || [])
    ?.map((part) => part?.text)
    ?.filter(Boolean)
    ?.join('\n')
    ?.trim();
}

function uniqueModels(models) {
  return [...new Set(models.filter(Boolean).map((m) => m.trim()).filter(Boolean))];
}

function isGeminiApiKeyError(statusCode, messageText) {
  const text = String(messageText || '').toLowerCase();

  if (statusCode === 401 || statusCode === 403) {
    return true;
  }

  return (
    text.includes('api key') ||
    text.includes('api_key') ||
    text.includes('credential') ||
    text.includes('permission denied') ||
    text.includes('unauthenticated')
  );
}

function buildLocalChatFallbackReply(message) {
  const prompt = String(message || '').trim();

  if (!prompt) {
    return 'Please share your farming question in one sentence. I can help with disease symptoms, prevention, irrigation, and treatment options.';
  }

  return [
    'Live AI chat is temporarily unavailable due to API key configuration, but here is practical guidance you can apply now:',
    '',
    `1. Clarify the issue: ${prompt}`,
    '2. Field check: inspect 10-15 plants across different spots and note symptom pattern (leaf spots, curling, wilting, stem lesions).',
    '3. Immediate containment: isolate severely affected plants and sanitize tools after each cut.',
    '4. Water management: avoid overhead irrigation late in the day; keep foliage dry where possible.',
    '5. Next step: if symptoms spread quickly, share crop type + stage + photos for a targeted treatment plan.',
  ].join('\n');
}

loadEnvFiles();

const MONGODB_URI = process.env.MONGODB_URI;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const PORT = parseInt(process.env.PORT || '8080', 10);
const JWT_SECRET = process.env.JWT_SECRET || 'cropsense_jwt_secret_change_in_production';
const DEFAULT_ALLOWED_ORIGINS = [
  'https://cropsenseaiapp.vercel.app',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
];

const ALLOWED_ORIGINS = [
  ...new Set(
    [...DEFAULT_ALLOWED_ORIGINS, ...(process.env.CORS_ORIGINS || '').split(',')]
      .map((origin) => origin.trim())
      .filter(Boolean)
  ),
];

const GEMINI_FALLBACK_MODELS = [
  GEMINI_MODEL,
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-1.5-flash-latest',
];

const app = express();

function getLocalMlDiagnostics() {
  const altModelDir = path.resolve(__dirname, 'ml_model', 'models');
  const defaultModelDir = path.resolve(__dirname, 'ml', 'models');
  const activeModelDir = fs.existsSync(altModelDir) ? altModelDir : defaultModelDir;
  const modelPath = path.join(activeModelDir, 'cropsense_model.pth');
  const classIndexPath = path.join(activeModelDir, 'class_index.json');
  const predictScriptPath = path.resolve(__dirname, 'ml', 'predict.py');

  return {
    pythonCandidates: getPythonCandidates(),
    isVercel: Boolean(process.env.VERCEL === '1' || process.env.VERCEL_ENV),
    activeModelDir,
    modelPath,
    modelExists: fs.existsSync(modelPath),
    classIndexPath,
    classIndexExists: fs.existsSync(classIndexPath),
    predictScriptPath,
    predictScriptExists: fs.existsSync(predictScriptPath),
  };
}

function getPythonCandidates() {
  const venvPath = path.resolve(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
  const venvUnixPath = path.resolve(__dirname, '..', '..', '.venv', 'bin', 'python3');
  
  return [
    ...new Set(
      [
        process.env.PYTHON_PATH,
        // Local virtual environment (highest priority)
        fs.existsSync(venvPath) ? venvPath : null,
        fs.existsSync(venvUnixPath) ? venvUnixPath : null,
        // System Python
        'python3',
        'python',
        'py',
        // Render deployment
        '/opt/render/project/python/bin/python3',
        '/opt/render/project/python/bin/python',
        // Unix/Linux
        '/usr/local/bin/python3',
        '/usr/bin/python3',
      ].filter(Boolean)
    ),
  ];
}

function sanitizeImageBase64(imageValue) {
  if (typeof imageValue !== 'string') {
    return '';
  }

  return imageValue.replace(/^data:\w+\/[-+.\w]+;base64,/, '').trim();
}

function getDecodedImageSize(imageBase64) {
  try {
    return Buffer.from(imageBase64, 'base64').byteLength;
  } catch {
    return 0;
  }
}

function runLocalPythonPredict(imageBase64) {
  const script = path.resolve(__dirname, 'ml', 'predict.py');
  const pythonCandidates = getPythonCandidates();
  const tempDir = path.resolve(__dirname, '.tmp-predict');

  fs.mkdirSync(tempDir, { recursive: true });

  return new Promise((resolve) => {
    const tempFile = path.join(tempDir, `predict-${Date.now()}-${crypto.randomUUID()}.bin`);
    fs.writeFileSync(tempFile, Buffer.from(imageBase64, 'base64'));

    const tryNext = (index, lastError = null) => {
      if (index >= pythonCandidates.length) {
        const detail = lastError?.message || 'No working Python executable found';
        fs.rmSync(tempFile, { force: true });
        return resolve({
          success: false,
          error: `Local prediction failed: ${detail}`,
        });
      }

      const python = pythonCandidates[index];
      console.log(`[ML Predict] Attempting Python execution: ${python} (${index + 1}/${pythonCandidates.length})`);
      
      execFile(
        python,
        [script, '--file', tempFile],
        {
          cwd: path.dirname(script),
          timeout: 30000,
          maxBuffer: 10 * 1024 * 1024,
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
        },
        (err, stdout, stderr) => {
          const stdoutText = stdout != null ? String(stdout).trim() : '';
          const stderrText = stderr != null ? String(stderr).trim() : '';

          const parseScriptResult = () => {
            if (!stdoutText) return null;
            try {
              const data = JSON.parse(stdoutText);
              data.pythonExecutable = python;
              return data;
            } catch {
              return null;
            }
          };

          const scriptResult = parseScriptResult();
          if (scriptResult) {
            console.log(`[ML Predict] Script returned JSON via ${python}:`, scriptResult);
            fs.rmSync(tempFile, { force: true });
            return resolve(scriptResult);
          }

          if (err) {
            console.error(`[ML Predict] Error via ${python} (Code: ${err.code}):`, stderrText || err.message, 'stdout:', stdoutText);
            return tryNext(index + 1, stderrText ? new Error(stderrText) : err);
          }

          console.error(`[ML Predict] Unexpected output via ${python}: stdout=${stdoutText}, stderr=${stderrText}`);
          return tryNext(index + 1, new Error('Unexpected Python output from local prediction'));        
        }
      );
    };

    return tryNext(0);
  });
}

function isAllowedOrigin(origin) {
  if (!origin) {
    return true;
  }

  if (ALLOWED_ORIGINS.includes(origin)) {
    return true;
  }

  try {
    const { hostname, protocol } = new URL(origin);
    const isLocal =
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname === '::1' ||
      hostname === '[::1]';

    if (isLocal) {
      return protocol === 'http:' || protocol === 'https:';
    }

    return protocol === 'https:' && hostname.endsWith('.vercel.app');
  } catch {
    return false;
  }
}

app.use(
  cors({
    origin(origin, callback) {
      if (isAllowedOrigin(origin)) {
        return callback(null, true);
      }

      return callback(new Error(`Origin ${origin} is not allowed by CORS.`));
    },
    credentials: true,
    methods: ['GET', 'POST', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);
app.use(express.json({ limit: '15mb' }));

// Request logging middleware for debugging
app.use((req, res, next) => {
  const start = Date.now();
  const requestId = `${req.method} ${req.path}`;
  console.log(`[${new Date().toISOString()}] → ${requestId}`);
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${new Date().toISOString()}] ← ${requestId} ${res.statusCode} (${duration}ms)`);
  });
  
  next();
});

function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7).trim() : null;

  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }

  try {
    req.user = jwt.verify(token, JWT_SECRET);
    return next();
  } catch {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

let dbClient;
let analysesColl;
let usersColl;
let dbReady = false;

async function initDb() {
  if (!MONGODB_URI) {
    console.warn('MONGODB_URI not set. Starting without database.');
    return;
  }

  try {
    console.log('Connecting to MongoDB...');
    dbClient = new MongoClient(MONGODB_URI, {
      serverSelectionTimeoutMS: 10000,
      connectTimeoutMS: 10000,
    });
    await dbClient.connect();
    const db = dbClient.db();
    analysesColl = db.collection('analyses');
    usersColl = db.collection('users');

    await usersColl.createIndex({ email: 1 }, { unique: true });

    dbReady = true;
    console.log('MongoDB connected');
  } catch (err) {
    console.error('MongoDB error:', err.message);
    dbReady = false;
  }
}

function ensureAnalysesReady(res) {
  if (!dbReady || !analysesColl) {
    res.status(503).json({ error: 'Database not ready.' });
    return false;
  }

  return true;
}

function ensureUsersReady(res) {
  if (!dbReady || !usersColl) {
    res.status(503).json({ error: 'Database not ready.' });
    return false;
  }

  return true;
}

function normalizeEmail(email) {
  return String(email || '').trim().toLowerCase();
}

function confidenceLabel(confidence) {
  if (typeof confidence === 'string') {
    const trimmed = confidence.trim();

    if (/^(high|medium|low)$/i.test(trimmed)) {
      return trimmed[0].toUpperCase() + trimmed.slice(1).toLowerCase();
    }

    const parsed = Number.parseFloat(trimmed);
    if (!Number.isNaN(parsed)) {
      return confidenceLabel(parsed);
    }

    return trimmed || 'Unknown';
  }

  if (typeof confidence === 'number' && Number.isFinite(confidence)) {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  }

  return 'Unknown';
}

function normalizeTextList(items) {
  if (!Array.isArray(items)) {
    return [];
  }

  return items.map((item) => String(item)).filter(Boolean);
}

const DISEASE_PROFILES = [
  {
    match: /healthy|normal|no disease|good health/i,
    severity: 'Low',
    description: 'The crop appears healthy with no obvious signs of disease or stress.',
    symptoms: ['Even leaf color', 'No visible lesions or discoloration', 'Normal growth pattern'],
    treatment: {
      organic: ['Continue current care practices', 'Maintain balanced irrigation and nutrition'],
      chemical: ['No chemical treatment required'],
    },
    prevention: ['Monitor plants regularly', 'Keep tools and growing area clean', 'Maintain proper spacing and airflow'],
  },
  {
    match: /blight/i,
    severity: 'High',
    description: 'Blight typically causes rapid tissue damage, browning, and spread under wet conditions.',
    symptoms: ['Dark or brown lesions', 'Rapid leaf collapse', 'Spreading spots after rain or irrigation'],
    treatment: {
      organic: ['Remove infected leaves immediately', 'Improve airflow around plants', 'Apply neem or copper-based organic spray where appropriate'],
      chemical: ['Use a labeled fungicide recommended for blight', 'Follow local agricultural extension guidance before spraying'],
    },
    prevention: ['Avoid overhead watering late in the day', 'Rotate crops each season', 'Disinfect pruning tools between plants'],
  },
  {
    match: /rust/i,
    severity: 'Medium',
    description: 'Rust diseases create powdery orange or brown pustules that weaken leaves over time.',
    symptoms: ['Rust-colored pustules', 'Yellowing around spots', 'Premature leaf drop'],
    treatment: {
      organic: ['Remove heavily infected leaves', 'Avoid overcrowding', 'Use sulfur or neem-based sprays if suitable for the crop'],
      chemical: ['Apply a crop-approved fungicide early', 'Repeat only according to label directions'],
    },
    prevention: ['Use resistant varieties when available', 'Space plants to improve drying', 'Avoid excess nitrogen'],
  },
  {
    match: /mildew/i,
    severity: 'Medium',
    description: 'Mildew forms powdery growth on leaves and spreads fast in humid conditions.',
    symptoms: ['White powder-like coating', 'Leaf curling', 'Reduced photosynthesis'],
    treatment: {
      organic: ['Trim affected foliage', 'Improve sunlight exposure', 'Apply organic sulfur or bicarbonate sprays where appropriate'],
      chemical: ['Use a fungicide labeled for mildew', 'Spray early in the infection cycle'],
    },
    prevention: ['Keep foliage dry where possible', 'Increase air circulation', 'Remove crop debris after harvest'],
  },
  {
    match: /rot/i,
    severity: 'High',
    description: 'Rot often affects roots, stems, or fruit and can quickly reduce plant vigor.',
    symptoms: ['Softened tissue', 'Dark, mushy areas', 'Wilting despite watering'],
    treatment: {
      organic: ['Improve drainage immediately', 'Remove damaged plant parts', 'Reduce watering frequency'],
      chemical: ['Use a crop-specific fungicide if recommended', 'Treat soil or seed only with approved products'],
    },
    prevention: ['Avoid waterlogging', 'Use clean seed and containers', 'Do not overwater'],
  },
  {
    match: /wilt/i,
    severity: 'High',
    description: 'Wilt usually affects the plant vascular system and causes drooping, even when soil is moist.',
    symptoms: ['Drooping leaves', 'Stunted growth', 'Discoloration of vascular tissue or stems'],
    treatment: {
      organic: ['Remove badly affected plants', 'Improve soil drainage and sanitation', 'Use disease-free planting material'],
      chemical: ['Apply a recommended soil treatment only if supported by local guidance', 'Follow all label restrictions'],
    },
    prevention: ['Rotate susceptible crops', 'Sterilize tools', 'Avoid planting in infected soil'],
  },
  {
    match: /spot/i,
    severity: 'Medium',
    description: 'Leaf spot diseases create localized lesions that may spread if humidity remains high.',
    symptoms: ['Circular or irregular spots', 'Yellow halos around lesions', 'Leaf drop in advanced cases'],
    treatment: {
      organic: ['Remove infected leaves', 'Avoid overhead irrigation', 'Use compost teas or biologicals only if proven for the crop'],
      chemical: ['Apply an approved fungicide when symptoms first appear', 'Repeat according to label instructions'],
    },
    prevention: ['Keep leaves dry', 'Use clean irrigation water', 'Clear fallen debris'],
  },
  {
    match: /curl|mosaic|virus/i,
    severity: 'High',
    description: 'Viral infections often cause curling, mottling, and stunted growth, and are typically spread by vectors.',
    symptoms: ['Curled or twisted leaves', 'Mosaic or mottled patterns', 'Reduced plant size and vigor'],
    treatment: {
      organic: ['Remove infected plants to limit spread', 'Control insect vectors', 'Use reflective mulch where appropriate'],
      chemical: ['Treat vector insects with a crop-approved insecticide if needed', 'There is usually no direct cure for viral infection'],
    },
    prevention: ['Use virus-free seeds or transplants', 'Control aphids and whiteflies', 'Rogue infected plants early'],
  },
  {
    match: /scab/i,
    severity: 'Medium',
    description: 'Scab produces roughened lesions on leaves, fruit, or tubers and can affect market quality.',
    symptoms: ['Rough or corky lesions', 'Surface blemishes', 'Reduced crop appearance or quality'],
    treatment: {
      organic: ['Improve airflow and sanitation', 'Remove infected debris', 'Apply crop-safe biological controls where available'],
      chemical: ['Use a fungicide labeled for scab control', 'Apply preventively when conditions favor disease'],
    },
    prevention: ['Rotate crops', 'Avoid overhead watering', 'Choose resistant varieties'],
  },
  {
    match: /anthracnose/i,
    severity: 'High',
    description: 'Anthracnose can attack leaves, stems, and fruit, leading to sunken lesions and decay.',
    symptoms: ['Sunken dark lesions', 'Fruit or leaf rot', 'Rapid spread in wet weather'],
    treatment: {
      organic: ['Prune infected tissue', 'Dispose of plant debris safely', 'Keep canopies dry'],
      chemical: ['Use a fungicide labeled for anthracnose', 'Follow pre-harvest intervals carefully'],
    },
    prevention: ['Improve ventilation', 'Avoid splashing water onto foliage', 'Use clean planting material'],
  },
];

function getDiseaseProfile(diseaseName) {
  const normalizedName = String(diseaseName || 'Unknown disease').trim() || 'Unknown disease';
  const profile = DISEASE_PROFILES.find((item) => item.match.test(normalizedName)) || {
    severity: 'Low',
    description: `Analysis completed for ${normalizedName}. Monitor the crop for changes and confirm with a local agronomist if symptoms persist.`,
    symptoms: ['Unclear symptom pattern', 'Requires field verification'],
    treatment: {
      organic: ['Remove obviously damaged tissue', 'Keep the crop under close observation'],
      chemical: ['Use crop-approved treatment only after confirming the diagnosis'],
    },
    prevention: ['Monitor regularly', 'Keep irrigation and field sanitation consistent', 'Confirm diagnosis with local expertise'],
  };

  return {
    disease_name: normalizedName,
    severity: profile.severity,
    description: profile.description,
    symptoms: profile.symptoms,
    treatment: profile.treatment,
    prevention: profile.prevention,
  };
}

function buildDiagnosisResponse(sourceResult, overrides = {}) {
  const diseaseName = String(
    overrides.disease_name || sourceResult?.disease_name || sourceResult?.disease || sourceResult?.plant_name || sourceResult?.name || 'Unknown disease'
  ).trim() || 'Unknown disease';
  const confidenceValue =
    typeof sourceResult?.confidence === 'number'
      ? sourceResult.confidence
      : Number.parseFloat(String(sourceResult?.confidence || ''));
  const confidence = overrides.confidence || confidenceLabel(sourceResult?.confidence);
  const profile = getDiseaseProfile(diseaseName);
  const sourceSymptoms = normalizeTextList(sourceResult?.symptoms);
  const sourceOrganic = normalizeTextList(sourceResult?.treatment?.organic);
  const sourceChemical = normalizeTextList(sourceResult?.treatment?.chemical);
  const sourcePrevention = normalizeTextList(sourceResult?.prevention);

  return {
    success: true,
    source: overrides.source || sourceResult?.source || 'local_ml',
    fallback: overrides.fallback ?? sourceResult?.fallback ?? false,
    thresholdUsed: overrides.thresholdUsed ?? sourceResult?.thresholdUsed ?? null,
    disease_name: profile.disease_name,
    confidence,
    confidenceValue: Number.isFinite(confidenceValue) ? confidenceValue : null,
    severity: overrides.severity || sourceResult?.severity || profile.severity,
    description: String(sourceResult?.description || profile.description),
    symptoms: sourceSymptoms.length > 0 ? sourceSymptoms : profile.symptoms,
    treatment: {
      organic: sourceOrganic.length > 0 ? sourceOrganic : profile.treatment.organic,
      chemical: sourceChemical.length > 0 ? sourceChemical : profile.treatment.chemical,
    },
    prevention: sourcePrevention.length > 0 ? sourcePrevention : profile.prevention,
    topPredictions: Array.isArray(sourceResult?.topPredictions) ? sourceResult.topPredictions : [],
    modelAccuracy: sourceResult?.modelAccuracy ?? null,
    note: sourceResult?.note ?? null,
    plantIdError: sourceResult?.plantIdError ?? null,
  };
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const derivedKey = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${derivedKey}`;
}

function verifyPassword(password, storedHash) {
  if (!storedHash || typeof storedHash !== 'string' || !storedHash.includes(':')) {
    return false;
  }

  const [salt, key] = storedHash.split(':');
  const derived = crypto.scryptSync(password, salt, 64);
  const original = Buffer.from(key, 'hex');

  if (derived.length !== original.length) {
    return false;
  }

  return crypto.timingSafeEqual(derived, original);
}

function sanitizeUser(user) {
  return {
    id: user._id?.toString?.() || user.id || '',
    email: user.email || '',
    name: user.name || '',
    picture: user.picture || null,
    role: user.role || 'user',
    provider: user.provider || 'local',
    scanCount: user.scanCount || 0,
    createdAt: user.createdAt || null,
  };
}

function signAuthToken(user) {
  return jwt.sign(
    {
      userId: user._id?.toString?.() || '',
      email: user.email,
      name: user.name,
      picture: user.picture || null,
      role: user.role || 'user',
    },
    JWT_SECRET,
    { expiresIn: '30d' }
  );
}

app.get('/', (req, res) =>
  res.json({
    ok: true,
    message: 'CropSense AI backend running.',
    health: '/api/health',
  })
);

// Debug endpoint to verify server routing is working
app.get('/api/debug', (req, res) =>
  res.json({
    ok: true,
    message: 'Debug endpoint working',
    timestamp: new Date().toISOString(),
    port: PORT,
    env: process.env.NODE_ENV || 'development',
    hasGeminiKey: Boolean(getGeminiApiKey()),
    dbReady,
  })
);

app.post('/api/auth/signup', async (req, res) => {
  try {
    if (!ensureUsersReady(res)) {
      return;
    }

    const name = String(req.body?.name || '').trim();
    const email = normalizeEmail(req.body?.email);
    const password = String(req.body?.password || '');

    if (!name) {
      return res.status(400).json({ error: 'Name is required.' });
    }

    if (!email || !email.includes('@')) {
      return res.status(400).json({ error: 'A valid email is required.' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters.' });
    }

    const existingUser = await usersColl.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ error: 'An account with this email already exists.' });
    }

    const user = {
      name,
      email,
      passwordHash: hashPassword(password),
      picture: null,
      role: 'user',
      provider: 'local',
      scanCount: 0,
      createdAt: new Date(),
      updatedAt: new Date(),
      lastLoginAt: new Date(),
    };

    const result = await usersColl.insertOne(user);
    user._id = result.insertedId;

    return res.json({
      ok: true,
      token: signAuthToken(user),
      user: sanitizeUser(user),
    });
  } catch (err) {
    console.error('Signup error:', err.message);
    return res.status(500).json({ error: 'Failed to create account.' });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    if (!ensureUsersReady(res)) {
      return;
    }

    const email = normalizeEmail(req.body?.email);
    const password = String(req.body?.password || '');

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required.' });
    }

    const user = await usersColl.findOne({ email });
    if (!user || !verifyPassword(password, user.passwordHash)) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    await usersColl.updateOne(
      { _id: user._id },
      { $set: { updatedAt: new Date(), lastLoginAt: new Date() } }
    );

    return res.json({
      ok: true,
      token: signAuthToken(user),
      user: sanitizeUser(user),
    });
  } catch (err) {
    console.error('Login error:', err.message);
    return res.status(500).json({ error: 'Failed to sign in.' });
  }
});

app.get('/api/auth/me', requireAuth, async (req, res) => {
  try {
    let user = req.user;

    if (dbReady && usersColl && req.user.userId) {
      const filters = [];

      if (ObjectId.isValid(req.user.userId)) {
        filters.push({ _id: new ObjectId(req.user.userId) });
      }

      if (filters.length > 0) {
        const dbUser = await usersColl.findOne(
          filters.length === 1 ? filters[0] : { $or: filters },
          {
            projection: {
              _id: 1,
              email: 1,
              name: 1,
              picture: 1,
              role: 1,
              provider: 1,
              scanCount: 1,
              createdAt: 1,
            },
          }
        );

        if (dbUser) {
          user = dbUser;
        }
      }
    }

    return res.json({ ok: true, user: sanitizeUser(user) });
  } catch (err) {
    console.error('Auth me error:', err.message);
    return res.status(500).json({ error: 'Failed to fetch user' });
  }
});

app.post('/api/auth/logout', (req, res) => {
  return res.json({ ok: true, message: 'Logged out successfully' });
});

app.get('/api/health', (req, res) =>
  res.json({
    ok: true,
    hasGeminiKey: Boolean(getGeminiApiKey()),
    dbReady,
    authEnabled: true,
    ml: getLocalMlDiagnostics(),
  })
);

app.get('/api/local_predict_health', (req, res) =>
  res.json({
    ok: true,
    ml: getLocalMlDiagnostics(),
  })
);

app.post('/api/analysis', async (req, res) => {
  const payload = req.body || {};

  try {
    if (!ensureAnalysesReady(res)) {
      return;
    }

    const doc = {
      timestamp: new Date(),
      summary: payload.summary || null,
      disease: payload.disease || null,
      diseaseDetails: payload.diseaseDetails || null,
      allDetected: Array.isArray(payload.allDetected) ? payload.allDetected : [],
      confidence: payload.confidence || null,
      severity: payload.severity || null,
      severityScore: payload.severityScore || 0,
      plantInfo: payload.plantInfo || null,
      isHealthy: payload.isHealthy === true,
      isDemo: payload.isDemo === true,
      imageBase64: payload.imageBase64 || null,
      isVerified: false,
      verifiedLabel: null,
      usedForTraining: false,
      meta: payload.meta || null,
    };

    const result = await analysesColl.insertOne(doc);
    return res.json({ ok: true, id: result.insertedId });
  } catch (err) {
    console.error('Insert error:', err);
    return res.status(500).json({ error: 'Failed to save analysis' });
  }
});

app.get('/api/analyses', async (req, res) => {
  try {
    if (!ensureAnalysesReady(res)) {
      return;
    }

    const limit = Math.min(parseInt(req.query.limit || '50', 10), 200);
    const skip = Math.max(parseInt(req.query.skip || '0', 10), 0);
    const filter = {};

    if (req.query.disease) {
      filter.disease = { $regex: req.query.disease, $options: 'i' };
    }

    if (req.query.isHealthy !== undefined) {
      filter.isHealthy = req.query.isHealthy === 'true';
    }

    if (req.query.isVerified !== undefined) {
      filter.isVerified = req.query.isVerified === 'true';
    }

    if (req.query.includeDemo !== 'true') {
      filter.isDemo = { $ne: true };
    }

    const [items, total, totalVerified, totalWithImage, diseaseCounts] = await Promise.all([
      analysesColl
        .find(filter, { projection: { imageBase64: 0 } })
        .sort({ timestamp: -1 })
        .skip(skip)
        .limit(limit)
        .toArray(),
      analysesColl.countDocuments(filter),
      analysesColl.countDocuments({ isVerified: true, isDemo: { $ne: true } }),
      analysesColl.countDocuments({ imageBase64: { $ne: null }, isDemo: { $ne: true } }),
      analysesColl.aggregate([
        { $match: { disease: { $ne: null }, isDemo: { $ne: true } } },
        { $group: { _id: '$disease', count: { $sum: 1 }, verified: { $sum: { $cond: ['$isVerified', 1, 0] } } } },
        { $sort: { count: -1 } },
        { $limit: 20 },
      ]).toArray(),
    ]);

    return res.json({
      ok: true,
      items,
      total,
      mlStats: {
        totalScans: total,
        totalVerified,
        totalWithImage,
        diseaseCounts,
      },
    });
  } catch (err) {
    console.error('Fetch analyses error:', err);
    return res.status(500).json({ error: 'Failed to fetch analyses' });
  }
});

app.patch('/api/analysis/:id', async (req, res) => {
  try {
    if (!ensureAnalysesReady(res)) {
      return;
    }

    const id = new ObjectId(req.params.id);
    const body = req.body || {};
    const update = {};

    if (body.verifiedLabel !== undefined) {
      update.verifiedLabel = body.verifiedLabel;
      update.isVerified = true;
    }

    if (body.isVerified === false) {
      update.isVerified = false;
      update.verifiedLabel = null;
    }

    if (Object.keys(update).length === 0) {
      return res.status(400).json({ error: 'No valid update fields provided' });
    }

    const result = await analysesColl.findOneAndUpdate(
      { _id: id },
      { $set: update },
      { returnDocument: 'after', projection: { imageBase64: 0 } }
    );

    if (!result) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    return res.json({ ok: true, item: result });
  } catch (err) {
    console.error('Update analysis error:', err);
    return res.status(500).json({ error: 'Failed to update analysis' });
  }
});

app.delete('/api/analysis/:id', async (req, res) => {
  try {
    if (!ensureAnalysesReady(res)) {
      return;
    }

    const id = new ObjectId(req.params.id);
    const result = await analysesColl.deleteOne({ _id: id });

    if (result.deletedCount === 0) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    return res.json({ ok: true, deleted: true });
  } catch (err) {
    console.error('Delete analysis error:', err);
    return res.status(500).json({ error: 'Failed to delete analysis' });
  }
});

app.get('/api/dataset', async (req, res) => {
  try {
    if (!ensureAnalysesReady(res)) {
      return;
    }

    const onlyVerified = req.query.onlyVerified !== 'false';
    const minConf = parseFloat(req.query.minConfidence || '0.5');
    const filter = {
      imageBase64: { $ne: null },
      isDemo: { $ne: true },
      confidence: { $gte: minConf },
    };

    if (onlyVerified) {
      filter.isVerified = true;
    }

    const records = await analysesColl.find(filter).toArray();
    const dataset = records
      .map((record) => ({
        id: record._id.toString(),
        label: record.verifiedLabel || record.disease,
        confidence: record.confidence,
        severityScore: record.severityScore || 0,
        imageBase64: record.imageBase64,
        createdAt: record.timestamp,
      }))
      .filter((record) => record.label && record.imageBase64);

    const classes = [...new Set(dataset.map((record) => record.label))].sort();

    if (records.length > 0) {
      await analysesColl.updateMany(
        { _id: { $in: records.map((record) => record._id) } },
        { $set: { usedForTraining: true } }
      );
    }

    return res.json({ ok: true, totalRecords: dataset.length, classes, dataset });
  } catch (err) {
    console.error('Dataset export error:', err);
    return res.status(500).json({ error: 'Failed to export dataset' });
  }
});

app.post('/api/chat', async (req, res) => {
  const { message, language, context } = req.body || {};
  const geminiApiKey = getGeminiApiKey();

  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'Message is required.' });
  }

  if (!geminiApiKey) {
    return res.json({
      ok: true,
      model: 'local-fallback',
      fallback: true,
      reply: buildLocalChatFallbackReply(message),
      warning: 'GEMINI_API_KEY is not configured on the server.',
    });
  }

  const systemPrompt = [
    'You are CropSense AI, an agricultural assistant for farmers.',
    'Give practical and safe farming guidance.',
    'Keep answers concise, structured, and easy to follow.',
    language ? `Reply in this language when possible: ${language}.` : null,
    context ? `Focus on this context: ${context}.` : null,
  ]
    .filter(Boolean)
    .join(' ');

  try {
    const modelsToTry = uniqueModels(GEMINI_FALLBACK_MODELS);
    let activeModel = null;
    let data = null;
    let lastError = null;

    for (const model of modelsToTry) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 25000);

      const geminiResponse = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${geminiApiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: systemPrompt }],
            },
            contents: [
              {
                role: 'user',
                parts: [{ text: message.trim() }],
              },
            ],
          }),
          signal: controller.signal,
        }
      ).finally(() => clearTimeout(timeout));

      data = await geminiResponse.json().catch(() => ({}));

      if (geminiResponse.ok) {
        activeModel = model;
        break;
      }

      const status = data?.error?.status || '';
      const messageText = data?.error?.message || '';
      const isMissingModel = geminiResponse.status === 404 || status === 'NOT_FOUND';

      if (isMissingModel) {
        console.warn(`Gemini model unavailable: ${model}. Trying next fallback...`);
        lastError = { status: geminiResponse.status, message: messageText };
        continue;
      }

      if (isGeminiApiKeyError(geminiResponse.status, messageText)) {
        console.warn('Gemini API key issue detected. Serving local chat fallback response.');
        return res.json({
          ok: true,
          model: 'local-fallback',
          fallback: true,
          reply: buildLocalChatFallbackReply(message),
          warning: messageText || 'Gemini API key is invalid or missing permissions.',
        });
      }

      return res.status(geminiResponse.status).json({
        error: messageText || 'Gemini request failed.',
      });
    }

    if (!activeModel) {
      const attempted = modelsToTry.join(', ');
      const detail = lastError?.message ? ` Last error: ${lastError.message}` : '';
      return res.status(500).json({
        error: `No available Gemini model found. Tried: ${attempted}.${detail}`,
      });
    }

    const reply = extractGeminiReply(data);
    return res.json({
      ok: true,
      model: activeModel,
      reply: reply || 'No response generated.',
    });
  } catch (err) {
    console.error('Chat route error:', err);

    if (err.name === 'AbortError') {
      return res.status(504).json({ error: 'Gemini request timed out. Please try again.' });
    }

    if (isGeminiApiKeyError(err?.status, err?.message)) {
      return res.json({
        ok: true,
        model: 'local-fallback',
        fallback: true,
        reply: buildLocalChatFallbackReply(message),
        warning: 'Gemini API key is invalid or unavailable.',
      });
    }

    return res.status(500).json({ error: 'Failed to contact Gemini.' });
  }
});

// Local ML prediction endpoint — runs the Python predictor with the uploaded base64 image.
app.post('/api/local_predict', async (req, res) => {
  try {
    const imageBase64 = sanitizeImageBase64(req.body?.imageBase64 || req.body?.image || '');
    const imageSizeBytes = getDecodedImageSize(imageBase64);

    if (!imageBase64 || typeof imageBase64 !== 'string' || imageBase64.trim() === '') {
      return res.status(400).json({ success: false, error: 'imageBase64 is required' });
    }

    if (!imageSizeBytes) {
      return res.status(400).json({ success: false, error: 'Invalid image encoding' });
    }

    // Enforce the limit using decoded bytes so frontend file-size checks match backend behavior.
    if (imageSizeBytes > 10 * 1024 * 1024) {
      return res.status(400).json({ success: false, error: 'Image must be under 10MB' });
    }

    const data = await runLocalPythonPredict(imageBase64);
    if (!data?.success) {
      return res.status(500).json(data);
    }
    return res.json(data);
  } catch (err) {
    console.error('Local predict endpoint error:', err.message);
    return res.status(500).json({ success: false, error: 'Local prediction endpoint error' });
  }
});

// Combined analysis endpoint: try local model first, then fallback to Plant.id API
app.post('/api/analyze_image', async (req, res) => {
  try {
    const imageBase64 = sanitizeImageBase64(req.body?.imageBase64 || req.body?.image || '');
    const imageSizeBytes = getDecodedImageSize(imageBase64);
    if (!imageBase64 || typeof imageBase64 !== 'string' || imageBase64.trim() === '') {
      return res.status(400).json({ success: false, error: 'imageBase64 is required' });
    }

    if (!imageSizeBytes) {
      return res.status(400).json({ success: false, error: 'Invalid image encoding' });
    }

    if (imageSizeBytes > 10 * 1024 * 1024) {
      return res.status(400).json({ success: false, error: 'Image must be under 10MB' });
    }

    const thresholdEnv = process.env.LOCAL_CONFIDENCE_THRESHOLD || '0.8';
    const threshold = Math.min(Math.max(parseFloat(thresholdEnv) || 0.8, 0), 1);

    // Detect if running on Vercel (serverless) or locally
    const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;
    const requestOrigin = `${req.protocol}://${req.get('host')}`;
    const mlPredictUrl =
      process.env.ML_PREDICT_URL ||
      (isVercel ? new URL('/api/ml_predict', requestOrigin).toString() : null);

    // Run the local predictor (local dev) or call serverless endpoint (Vercel)
    const runLocalPredict = async () => {
      // On Vercel: call the Python serverless endpoint
      if (isVercel && mlPredictUrl) {
        try {
          const mlResponse = await fetch(`${mlPredictUrl}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageBase64 }),
          });
          if (mlResponse.ok) {
            return await mlResponse.json();
          } else {
            const text = await mlResponse.text();
            console.error('ML endpoint error:', mlResponse.status, text);
            return { success: false, error: 'ML prediction service unavailable' };
          }
        } catch (err) {
          console.error('ML endpoint request failed:', err.message);
          return { success: false, error: 'ML prediction service unavailable' };
        }
      }

      // Local development: use execFile to run predict.py
      return runLocalPythonPredict(imageBase64);
    };

    const localResult = await runLocalPredict();

    // If local model returns a confident prediction above threshold, return it
    if (
      localResult &&
      localResult.success === true &&
      localResult.disease &&
      typeof localResult.confidence === 'number' &&
      localResult.confidence >= threshold
    ) {
      localResult.fallback = false;
      localResult.thresholdUsed = threshold;
      return res.json(buildDiagnosisResponse(localResult, { source: 'local_ml', fallback: false, thresholdUsed: threshold }));
    }

    // Otherwise, attempt Plant.id fallback if API key is configured
    const plantKey = process.env.PLANT_ID_API_KEY;
    if (!plantKey) {
      // If no Plant.id key, return local result (even if low confidence) with a note
      const fallback = localResult || { success: false, error: 'No local prediction available' };
      fallback.fallback = true;
      fallback.note = 'No PLANT_ID_API_KEY configured for external analysis';
      fallback.thresholdUsed = threshold;
      return res.json(buildDiagnosisResponse(fallback, { source: fallback.source || 'local_ml', fallback: true, thresholdUsed: threshold }));
    }

    // Prepare base64 without data URI prefix
    const cleanBase64 = imageBase64.replace(/^data:\w+\/[-+.\w]+;base64,/, '');

    const plantResp = await fetch('https://api.plant.id/v2/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: plantKey, images: [cleanBase64] }),
    }).catch((err) => {
      console.error('Plant.id request failed:', err?.message || err);
      return null;
    });

    if (!plantResp || !plantResp.ok) {
      const statusText = plantResp ? `${plantResp.status} ${plantResp.statusText}` : 'No response';
      console.error('Plant.id API error:', statusText);
      const fallback = localResult || { success: false, error: 'External analysis failed' };
      fallback.fallback = true;
      fallback.thresholdUsed = threshold;
      fallback.plantIdError = statusText;
      return res.json(fallback);
    }

    const plantData = await plantResp.json().catch(() => ({}));

    // Extract top suggestion and probability if available
    const top = (plantData.suggestions && plantData.suggestions[0]) || null;
    const plantName = top?.plant_name || top?.name || null;
    const plantProb = top?.probability || (top?.probability?.toFixed ? top.probability : undefined) || 0;

    const out = {
      success: true,
      source: 'plant_id',
      plantIdRaw: plantData,
      disease: plantName || null,
      confidence: typeof plantProb === 'number' ? plantProb : parseFloat(plantProb) || 0,
      fallback: true,
      thresholdUsed: threshold,
    };

    return res.json(buildDiagnosisResponse(out, { source: 'plant_id', fallback: true, thresholdUsed: threshold }));
  } catch (err) {
    console.error('Analyze image route error:', err);
    return res.status(500).json({ success: false, error: 'Analyze image failed' });
  }
});

// Startup diagnostics
function logStartupDiagnostics() {
  console.log('\n====== CropSense AI Backend Startup ======');
  console.log(`Port: ${PORT}`);
  console.log(`Node version: ${process.version}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  const mlDiags = getLocalMlDiagnostics();
  console.log('\n--- ML Configuration ---');
  console.log(`Active model directory: ${mlDiags.activeModelDir}`);
  console.log(`Model file exists: ${mlDiags.modelExists}`);
  console.log(`Class index exists: ${mlDiags.classIndexExists}`);
  console.log(`Python script path: ${mlDiags.predictScriptPath}`);
  console.log(`Python candidates: ${mlDiags.pythonCandidates.join(', ')}`);
  console.log(`Running on Render: ${mlDiags.isVercel ? 'Yes' : 'No'}`);
  
  // Quick Python verification
  for (const python of mlDiags.pythonCandidates.slice(0, 3)) {
    try {
      const version = execSync(`${python} --version`, { encoding: 'utf-8', timeout: 5000 }).trim();
      console.log(`✓ Python available: ${python} (${version})`);
      
      // Check torch
      try {
        execSync(`${python} -c "import torch; print('PyTorch ' + torch.__version__)"`, { 
          encoding: 'utf-8', 
          timeout: 10000,
          stdio: 'pipe'
        });
        console.log(`✓ PyTorch is available`);
      } catch {
        console.warn(`✗ PyTorch not available in ${python}`);
      }
      break;
    } catch (err) {
      console.warn(`✗ Python not available: ${python}`);
    }
  }
  
  console.log('=====================================\n');
}

// Silently handle development-only requests (no logging to keep logs clean)
app.get('/ws', (req, res) => res.status(404).end());
app.get('/manifest.json', (req, res) => res.status(404).end());

// 404 handler for undefined API routes - should return JSON not HTML
app.use((req, res) => {
  console.warn(`404 - ${req.method} ${req.path}`);
  res.status(404).json({
    ok: false,
    error: 'Endpoint not found',
    method: req.method,
    path: req.path,
    availableEndpoints: [
      'POST /api/auth/signup',
      'POST /api/auth/login',
      'GET /api/auth/me',
      'POST /api/auth/logout',
      'GET /api/health',
      'POST /api/analysis',
      'GET /api/analyses',
      'PATCH /api/analysis/:id',
      'DELETE /api/analysis/:id',
      'GET /api/dataset',
      'POST /api/chat',
      'POST /api/local_predict',
      'POST /api/analyze_image',
    ],
  });
});

app.listen(PORT, () => {
  console.log(`CropSense backend running on port ${PORT}`);
  logStartupDiagnostics();
});
initDb().catch((err) => console.error('initDb failed:', err));

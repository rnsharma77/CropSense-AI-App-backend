const path = require('path');
const crypto = require('crypto');
const dotenv = require('dotenv');
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const { MongoClient, ObjectId } = require('mongodb');
const { execFile } = require('child_process');

const ENV_PATHS = [
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

loadEnvFiles();

const MONGODB_URI = process.env.MONGODB_URI;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const PORT = process.env.PORT || 5050;
const JWT_SECRET = process.env.JWT_SECRET || 'cropsense_jwt_secret_change_in_production';
const DEFAULT_ALLOWED_ORIGINS = [
  'https://crop-sense-ai-app.vercel.app',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
];

const ALLOWED_ORIGINS = (process.env.CORS_ORIGINS || DEFAULT_ALLOWED_ORIGINS.join(','))
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

const GEMINI_FALLBACK_MODELS = [
  GEMINI_MODEL,
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-1.5-flash-latest',
];

const app = express();

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
    return res.status(500).json({ error: 'GEMINI_API_KEY is not configured on the server.' });
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

    return res.status(500).json({ error: 'Failed to contact Gemini.' });
  }
});

// Local ML prediction endpoint — runs the Python predictor with the uploaded base64 image.
app.post('/api/local_predict', async (req, res) => {
  try {
    const imageBase64 = req.body?.imageBase64 || req.body?.image || '';

    if (!imageBase64 || typeof imageBase64 !== 'string' || imageBase64.trim() === '') {
      return res.status(400).json({ success: false, error: 'imageBase64 is required' });
    }

    // Safety: limit size (example: ~10MB base64) to avoid abuse
    if (imageBase64.length > 10 * 1024 * 1024) {
      return res.status(400).json({ success: false, error: 'Image too large' });
    }

    const python = process.env.PYTHON_PATH || 'python';
    const script = path.resolve(__dirname, 'ml', 'predict.py');

    execFile(
      python,
      [script, '--image', imageBase64],
      { timeout: 30000, maxBuffer: 10 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          console.error('Local predict error:', err, stderr?.toString?.());
          return res.status(500).json({ success: false, error: 'Local prediction failed' });
        }

        try {
          const data = JSON.parse(stdout || '{}');
          return res.json(data);
        } catch (e) {
          console.error('Local predict parse error:', e, 'stdout:', stdout);
          return res.status(500).json({ success: false, error: 'Invalid predictor output' });
        }
      }
    );
  } catch (err) {
    console.error('Local predict endpoint error:', err.message);
    return res.status(500).json({ success: false, error: 'Local prediction endpoint error' });
  }
});

// Combined analysis endpoint: try local model first, then fallback to Plant.id API
app.post('/api/analyze_image', async (req, res) => {
  try {
    const imageBase64 = req.body?.imageBase64 || req.body?.image || '';
    if (!imageBase64 || typeof imageBase64 !== 'string' || imageBase64.trim() === '') {
      return res.status(400).json({ success: false, error: 'imageBase64 is required' });
    }

    if (imageBase64.length > 10 * 1024 * 1024) {
      return res.status(400).json({ success: false, error: 'Image too large' });
    }

    const thresholdEnv = process.env.LOCAL_CONFIDENCE_THRESHOLD || '0.8';
    const threshold = Math.min(Math.max(parseFloat(thresholdEnv) || 0.8, 0), 1);

    const python = process.env.PYTHON_PATH || 'python';
    const script = path.resolve(__dirname, 'ml', 'predict.py');

    // Run the local predictor
    const runLocalPredict = () =>
      new Promise((resolve) => {
        execFile(
          python,
          [script, '--image', imageBase64],
          { timeout: 30000, maxBuffer: 10 * 1024 * 1024 },
          (err, stdout, stderr) => {
            if (err) {
              console.error('Local predict error (analyze_image):', err, stderr?.toString?.());
              return resolve({ success: false, error: 'Local prediction failed' });
            }

            try {
              const data = JSON.parse(stdout || '{}');
              return resolve(data);
            } catch (e) {
              console.error('Local predict parse error (analyze_image):', e, 'stdout:', stdout);
              return resolve({ success: false, error: 'Invalid predictor output' });
            }
          }
        );
      });

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
      return res.json(localResult);
    }

    // Otherwise, attempt Plant.id fallback if API key is configured
    const plantKey = process.env.PLANT_ID_API_KEY;
    if (!plantKey) {
      // If no Plant.id key, return local result (even if low confidence) with a note
      const fallback = localResult || { success: false, error: 'No local prediction available' };
      fallback.fallback = true;
      fallback.note = 'No PLANT_ID_API_KEY configured for external analysis';
      fallback.thresholdUsed = threshold;
      return res.json(fallback);
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

    return res.json(out);
  } catch (err) {
    console.error('Analyze image route error:', err);
    return res.status(500).json({ success: false, error: 'Analyze image failed' });
  }
});

app.listen(PORT, () => console.log(`CropSense backend running on port ${PORT}`));
initDb().catch((err) => console.error('initDb failed:', err));

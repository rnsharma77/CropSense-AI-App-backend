const path = require("path");
const dotenv = require("dotenv");

const ENV_PATHS = [
  path.resolve(__dirname, "../.env.local"),
  path.resolve(__dirname, "../.env"),
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
const express = require('express');
const cors = require('cors');
const { MongoClient, ObjectId } = require('mongodb');

const MONGODB_URI = process.env.MONGODB_URI;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const PORT = process.env.PORT || 5050;
const DEFAULT_ALLOWED_ORIGINS = [
  'https://crop-sense-ai-app.vercel.app',
  'http://localhost:3000',
  'http://127.0.0.1:3000',
];
const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS ||
  DEFAULT_ALLOWED_ORIGINS.join(',')
)
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
app.use(
  cors({
    origin(origin, callback) {
      if (!origin || ALLOWED_ORIGINS.includes(origin)) {
        return callback(null, true);
      }

      return callback(new Error(`Origin ${origin} is not allowed by CORS.`));
    },
  })
);
app.use(express.json({ limit: '5mb' }));

app.get('/', (req, res) =>
  res.json({
    ok: true,
    message: 'CropSense AI backend is running.',
    health: '/api/health',
  })
);

let dbClient;
let analysesColl;
let dbReady = false;

async function initDb() {
  if (!MONGODB_URI) {
    console.warn('MONGODB_URI not set. Starting server without database features.');
    return;
  }

  try {
    console.log('Attempting MongoDB connection...');
    dbClient = new MongoClient(MONGODB_URI, {
      serverSelectionTimeoutMS: 10000,
      connectTimeoutMS: 10000,
    });
    await dbClient.connect();
    const db = dbClient.db();
    analysesColl = db.collection('analyses');
    dbReady = true;
    console.log('✓ Connected to MongoDB successfully');
  } catch (err) {
    console.error('✗ MongoDB connection error:', err.message);
    console.error('Please ensure:');
    console.error('  1. MongoDB URI is correct in .env');
    console.error('  2. MongoDB Atlas cluster is active');
    console.error('  3. Your IP is whitelisted in MongoDB Atlas');
    dbReady = false;
  }
}

app.post('/api/analysis', async (req, res) => {
  const payload = req.body || {};
  try {
    if (!dbReady || !analysesColl) {
      return res.status(503).json({ error: 'Database not ready. Please try again in a moment.' });
    }

    const doc = {
      timestamp: new Date(),
      summary: payload.summary || null,
      disease: payload.disease || null,
      confidence: payload.confidence || null,
      plantInfo: payload.plantInfo || null,
      isDemo: payload.isDemo === true,
      meta: payload.meta || null,
    };

    const result = await analysesColl.insertOne(doc);
    return res.json({ ok: true, id: result.insertedId });
  } catch (err) {
    console.error('Insert error:', err);
    return res.status(500).json({ error: 'Failed to save analysis' });
  }
});

app.get('/api/health', (req, res) =>
  res.json({
    ok: true,
    hasGeminiKey: Boolean(getGeminiApiKey()),
  })
);

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
          headers: {
            'Content-Type': 'application/json',
          },
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

      console.error('Gemini API error:', data);
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

// List analyses with optional limit and skip
app.get('/api/analyses', async (req, res) => {
  try {
    if (!dbReady || !analysesColl) {
      return res.status(503).json({ error: 'Database not ready. Please try again in a moment.' });
    }
    const limit = Math.min(parseInt(req.query.limit || '50', 10), 200);
    const skip = Math.max(parseInt(req.query.skip || '0', 10), 0);
    const cursor = analysesColl.find({}).sort({ timestamp: -1 }).skip(skip).limit(limit);
    const items = await cursor.toArray();
    return res.json({ ok: true, items });
  } catch (err) {
    console.error('Fetch analyses error:', err);
    return res.status(500).json({ error: 'Failed to fetch analyses' });
  }
});

// Delete a single analysis by ID
app.delete('/api/analysis/:id', async (req, res) => {
  try {
    if (!analysesColl) return res.status(500).json({ error: 'Database not initialized' });
    const { ObjectId } = require('mongodb');
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

app.listen(PORT, () => console.log(`Server listening ${PORT}`));
initDb().catch((err) => {
  console.error('Init DB failed:', err);
});

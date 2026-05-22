const fs = require("fs");
const path = require("path");
const dotenv = require("../node_modules/dotenv");
const { MongoClient } = require("../node_modules/mongodb");

dotenv.config({ path: path.resolve(__dirname, "../../.env.local") });
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const mongoUri = process.env.MONGODB_URI;
const outputPath = path.resolve(__dirname, "models", "dataset_export.json");
const minConfidence = Number(process.env.ML_MIN_CONFIDENCE || 0.5);

async function main() {
  if (!mongoUri) {
    throw new Error("MONGODB_URI is not configured.");
  }

  const client = new MongoClient(mongoUri, {
    serverSelectionTimeoutMS: 15000,
    connectTimeoutMS: 15000,
  });

  try {
    await client.connect();
    const collection = client.db().collection("analyses");
    const records = await collection
      .find(
        {
          imageBase64: { $ne: null },
          isDemo: { $ne: true },
          isVerified: true,
          confidence: { $gte: minConfidence },
        },
        {
          projection: {
            verifiedLabel: 1,
            disease: 1,
            confidence: 1,
            severityScore: 1,
            imageBase64: 1,
            timestamp: 1,
          },
        }
      )
      .toArray();

    const dataset = records
      .map((row) => ({
        id: String(row._id),
        label: row.verifiedLabel || row.disease,
        confidence: row.confidence,
        severityScore: row.severityScore || 0,
        imageBase64: row.imageBase64,
        createdAt: row.timestamp,
      }))
      .filter((row) => row.label && row.imageBase64);

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(
      outputPath,
      JSON.stringify(
        {
          ok: true,
          totalRecords: dataset.length,
          classes: [...new Set(dataset.map((row) => row.label))].sort(),
          dataset,
        },
        null,
        2
      ),
      "utf-8"
    );

    console.log(`Exported ${dataset.length} records to ${outputPath}`);
  } finally {
    await client.close().catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});

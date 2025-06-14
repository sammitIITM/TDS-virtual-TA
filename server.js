// server.js
import express from "express";
import bodyParser from "body-parser";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAI } from "openai";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

if (
  !process.env.OPENAI_API_KEY ||
  !process.env.PINECONE_API_KEY ||
  !process.env.PINECONE_ENVIRONMENT ||
  !process.env.PINECONE_INDEX_NAME
) {
  console.error(
    "Missing OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, or PINECONE_INDEX_NAME in .env"
  );
  process.exit(1);
}

const openai = new OpenAI();

const pine = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineIndex = pine.Index(process.env.PINECONE_INDEX_NAME);

async function getEmbedding(text) {
  const resp = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  console.log(resp);
  return resp.data[0]?.embedding;
}

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "100mb" }));

app.get("/", async (req, res) => {
  res.json({ message: "server is running" });
});

app.post("/api", async (req, res) => {
  try {
    const { question, image } = req.body;
    if (!question) {
      return res.status(400).json({ error: "`question` is required" });
    }

    // 1) Optionally OCR the image → append to question (LEFT as TODO)
    let fullQuery = question;
    if (image) {
      // TODO: decode base64 & run OCR, e.g. via tesseract.js
      // const ocrText = await runOCR(image);
      // fullQuery += '\n\n[Image text]\n' + ocrText;
    }

    const qEmbedding = await getEmbedding(fullQuery);

    const queryResp = await pineIndex.query({
      vector: qEmbedding,
      topK: 15,
      includeMetadata: true,
    });

    const matches = queryResp.matches || [];
    const links = matches.map((m) => ({
      url: m.metadata.url,
      text: (m.metadata.text || "").slice(0, 100).replace(/\s+/g, " ") + "…",
    }));

    let prompt = `
You are a virtual TA for the IITM “Tools in Data Science” course.
Use only the following excerpts (with their URLs) to answer the student’s question.  
If the answer is not contained here, say “I’m not sure; please check the course materials or ask on Discourse.”  

Excerpts:
${matches
  .map((m, i) => `— [${i + 1}] ${m.metadata.url}\n${m.metadata.text}\n`)
  .join("\n")}

Student question: """${question}"""
Answer concisely and include an array of “links” (each with url and a one-line description).
`;

    const chatResp = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: "You are a helpful TA." },
        { role: "user", content: prompt },
      ],
      temperature: 0.2,
      max_tokens: 512,
    });

    const answerText = chatResp.choices[0].message.content.trim();
    return res.json({ answer: answerText, links });
  } catch (err) {
    console.error("API error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 API server listening on http://localhost:${PORT}/api`);
});

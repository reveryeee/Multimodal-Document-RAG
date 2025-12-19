# Multimodal-Document-RAG
This is a simple multimodal RAG pipeline for pre-sales&after-sales parts of logistics&warehousing automation industry. 

# Background and Idea
In the industries of logistics&warehousing automation&mobile robots, pre-sales and post-sales personnel frequently need to search for product technical specifications, material codes, error report solutions, and so on across numerous internal corporate databases spanning multiple departments during their daily work. During peak periods, manually retrieving large volumes of data is extremely time-consuming. This project aims to build a multimodal Retrieval-Augmented Generation (RAG) pipeline based on Langchain to enhance information retrieval and processing efficiency.

# Test Knowledge Base&Data
Our test knowledge base consists of the manual books of WMS system, WCS system, and AGV fleet management system.

# Preprocessing/Parsing
We use Camelot for structured table processing, pyMuPDF for image parsing, and pdfplumber for extracting text from each page. (Note: Calls to MinerU on the cloud platform have consistently failed. Once such issues are resolved in the future, the preprocessing/parsing steps of our multimodal content will be completed entirely using only MinerU.)

# Chunk
For multimodal content, we have both single-modal chunks and chunks integrated by different modals. Specifically, the chunk types include pure text chunk(Split by paragraph), pure table chunk, pure image chunk, text + image chunk (When both images and relevant text appear on the same page), text + table chunk(When both table and relevant text appear on the same page).    Reference link: https://blog.milvus.io/ai-quick-reference/what-are-effective-chunking-strategies-for-multimodal-documents

# Embedding Regarding Different Chunk Types
1. Text chunk → Encode with SentenceTransformers;  2. Table chunk → Encode with SentenceTransformers (after text structuring);  3. Image → CLIP; 4. Text + table chunk → SentenceTransformers (Encode them after assembling the text);  5. Text + Image → Text ST + CLIP (we conduct the dual encoding, and then vector concatenation).    

# Vector Store
Regarding the issue of mismatched dimensions in different vector spaces (since we have different chunk types and after embedding we have different vector space dimension), we create multiple FAISS indexes, each corresponding to one vector dimension.

# Generation
LLM: Qwen2.5-VL-7B-Instruct.  

# Evaluation Plan
Generate Q&A dataset with Easy dataset + MinerU + Qwen3-VL-8B-Thinking, and then manually verify and correct each Q&A pair based on the original document content to obtain the final ground truth dataset. Evaluate metrics could be Faithfulness, Answer Relevancy, Context Precision, Context Recall, and evaluation tools could be Ragas.

# The Future Expansion Plans for the Project
Build a pipeline for an Adaptive RAG or an agent capable of making decision autonomously. Such frameworks must possess the capability to determine which path the system should take based on different user query types before the LLM generates a response. For instance, when a user query pertains to internal product parameters, the system should follow the RAG path to retrieve internal documentation. For business opportunity exploration queries, it should follow the MCP function calling path to search external information sources. When a query is unrelated to the strict business operations, the system could follow the LLM direct response path.

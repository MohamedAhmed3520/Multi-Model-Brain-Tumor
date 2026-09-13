# Tumor Agent RAG

This repository provides a production-oriented but lightweight multimodal medical AI RAG + multi-agent skeleton built around the preserved model artifacts:

- `Tumor Models/best_model.h5` for the TensorFlow/Keras InceptionV3 classification model.
- `Tumor Models/best.pt` for the existing detection checkpoint.
- `Tumor Models/my_checkpoint.pth` for the existing PyTorch U-Net checkpoint.

The architecture separates concerns across:

- `app/models/` model service wrappers
- `app/agents/` CrewAI-style orchestrator and tools
- `app/agent/` LangGraph-style state, router, nodes, checkpoints, and graph
- `app/rag/` LangChain ingestion and retrieval helpers
- `app/search/` web evidence helper
- `app/database/` SQLAlchemy repositories and metadata model records
- `app/mcp/` FastMCP server interface
- `app/schemas/` Pydantic model schemas

The folders are intentionally scaffolded without changing the existing trained model files.

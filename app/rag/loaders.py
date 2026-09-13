from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader


class RAGLoaderFactory:
    @staticmethod
    def load(path: str):
        suffix = Path(path).suffix.lower()
        if suffix == '.txt':
            return TextLoader(path)
        if suffix == '.pdf':
            return PyPDFLoader(path)
        if suffix == '.docx':
            return Docx2txtLoader(path)
        if suffix in {'.html', '.htm'}:
            return UnstructuredHTMLLoader(path)
        return TextLoader(path)

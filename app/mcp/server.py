from fastmcp import FastMCP

mcp = FastMCP('medical-rag-agent')


@mcp.tool()
def analyze_image_classification(image_path: str):
    return {'status': 'not_implemented', 'image_path': image_path}


@mcp.tool()
def run_detection(image_path: str):
    return {'status': 'not_implemented', 'image_path': image_path}


@mcp.tool()
def run_segmentation(image_path: str):
    return {'status': 'not_implemented', 'image_path': image_path}


@mcp.tool()
def transcribe_audio(audio_path: str):
    return {'status': 'not_implemented', 'audio_path': audio_path}


@mcp.tool()
def search_knowledge_base(query: str):
    return {'status': 'not_implemented', 'query': query}


@mcp.tool()
def search_web(query: str):
    return {'status': 'not_implemented', 'query': query}


@mcp.tool()
def analyze_multimodal_case(query: str, image_paths: list[str] | None = None, audio_paths: list[str] | None = None):
    return {'status': 'not_implemented', 'query': query, 'image_paths': image_paths or [], 'audio_paths': audio_paths or []}

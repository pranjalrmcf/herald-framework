# Autonomous Research Analyst

A production-grade autonomous research analyst system using agent-controlled RAG architecture with graph reasoning.

## Features

- **Input Guardrails**: Safety, scope, and prompt injection checks
- **Query Processing**: Normalization, intent classification, and complexity estimation
- **Intelligent Routing**: Decides between fast vector search and research-grade graph construction
- **Hybrid Retrieval**: Combines web search, vector embeddings, and knowledge graph reasoning
- **Graph RAG**: Extracts entities and relationships for multi-hop reasoning
- **Quality Evaluation**: Automatic quality metrics with self-healing capabilities
- **Multi-layer Caching**: Optimized for cost and latency

## Project Structure

```
research_analyst/
├── config/              # Configuration and settings
├── core/                # Core models and exceptions
├── guardrails/          # Input and output validation
├── query_processing/    # Query normalization and classification
├── routing/             # Execution path routing
├── retrieval/           # Web search and vector retrieval
├── graph/               # Knowledge graph construction and querying
├── synthesis/           # Answer generation and context building
├── evaluation/          # Quality metrics and self-healing
├── caching/             # Multi-layer caching
├── orchestration/       # Pipeline orchestration
├── utils/               # Utilities and logging
├── api/                 # REST API
└── tests/               # Unit and integration tests
```

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_trf
   ```
5. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit `.env` file with your API keys and preferences:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `DEFAULT_LLM_PROVIDER`: Choose "openai" or "anthropic"
- See `.env.example` for all configuration options

## Usage

### Command Line
```bash
python -m research_analyst.cli "What is the relationship between OpenAI and Microsoft?"
```

### API Server
```bash
python -m research_analyst.api.main
```

Then access: `http://localhost:8000/docs`

### Python API
```python
from research_analyst import ResearchAnalyst

analyst = ResearchAnalyst()
result = analyst.query("What are the latest developments in quantum computing?")
print(result.answer.text)
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black research_analyst/
flake8 research_analyst/
```

## Architecture

The system uses a modular pipeline architecture:

1. **Input** → Guardrails (safety checks)
2. **Query Processing** → Normalization + Intent Classification
3. **Routing** → Fast Path (vector) or Research Path (graph)
4. **Retrieval** → Web search + Document processing
5. **Graph Construction** (if research path) → Entity/Relationship extraction
6. **Synthesis** → Context building + Answer generation
7. **Evaluation** → Quality metrics + Self-healing (if needed)
8. **Output** → Guardrails + Response

## License

MIT License

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Support

For issues and questions, please open a GitHub issue.

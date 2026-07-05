# Groq Mirror Professional

A high-performance, production-grade web interface for Groq API with advanced streaming capabilities, Markdown support, and stable Mermaid.js diagram rendering.

## Features

- **High-Performance Streaming**: Uses `ReadableStream` and `requestAnimationFrame` for stutter-free UI updates.
- **Incremental Rendering**: Intelligent Markdown parsing that handles incomplete blocks during streaming.
- **Stable Mermaid.js**: Renders diagrams only when complete, using `mermaid.render()` with caching to prevent flickering.
- **Optimized DOM**: Uses a Render Scheduler to prevent CPU spikes and Memory Leaks.
- **Professional UI**: Dark theme, responsive design, and syntax highlighting.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
/**
 * Groq Mirror Professional - Core Streaming Engine
 * Designed for high performance, incremental rendering, and stable Mermaid integration.
 */

class RenderScheduler {
    constructor() {
        this.queue = [];
        this.isProcessing = false;
    }

    enqueue(task) {
        this.queue.push(task);
        if (!this.isProcessing) {
            this.process();
        }
    }

    process() {
        if (this.queue.length === 0) {
            this.isProcessing = false;
            return;
        }

        this.isProcessing = true;
        requestAnimationFrame(() => {
            const task = this.queue.shift();
            task();
            this.process();
        });
    }
}

class MermaidManager {
    constructor() {
        this.cache = new Map();
        this.counter = 0;
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
        });
    }

    async renderDiagram(containerId, code) {
        if (this.cache.get(containerId) === code) return;

        const container = document.getElementById(containerId);
        if (!container) return;

        try {
            const id = `mermaid-${Date.now()}-${this.counter++}`;
            const { svg } = await mermaid.render(id, code);
            container.innerHTML = svg;
            this.cache.set(containerId, code);
        } catch (err) {
            console.error("Mermaid Render Error:", err);
            container.innerHTML = `<div class="error">Invalid Mermaid Syntax</div>`;
        }
    }
}

class StreamingEngine {
    constructor() {
        this.chatContainer = document.getElementById('chat-container');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        
        this.scheduler = new RenderScheduler();
        this.mermaidManager = new MermaidManager();
        this.abortController = null;
        this.messages = [];
        
        this.setupMarked();
        this.initEventListeners();
    }

    setupMarked() {
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true
        });
    }

    initEventListeners() {
        this.sendBtn.addEventListener('click', () => this.handleSend());
        this.stopBtn.addEventListener('click', () => this.handleStop());
        this.clearBtn.addEventListener('click', () => this.clearChat());
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSend();
            }
        });
    }

    async handleSend() {
        const text = this.userInput.value.trim();
        if (!text) return;

        this.userInput.value = '';
        this.addMessage(text, 'user');
        
        this.messages.push({ role: 'user', content: text });
        await this.startStreaming();
    }

    handleStop() {
        if (this.abortController) {
            this.abortController.abort();
            this.finalizeStream();
        }
    }

    clearChat() {
        this.chatContainer.innerHTML = '';
        this.messages = [];
    }

    addMessage(text, role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message markdown-body`;
        
        // Use DocumentFragment for performance
        const fragment = document.createDocumentFragment();
        fragment.appendChild(msgDiv);
        this.chatContainer.appendChild(fragment);
        
        if (role === 'user') {
            msgDiv.textContent = text;
        }
        
        this.scrollToBottom();
        return msgDiv;
    }

    async startStreaming() {
        this.toggleLoading(true);
        this.abortController = new AbortController();
        
        const botMsgDiv = this.addMessage('', 'bot');
        const contentSpan = document.createElement('div');
        botMsgDiv.appendChild(contentSpan);

        let fullText = "";
        let buffer = "";
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: this.messages }),
                signal: this.abortController.signal
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    responseBuffer += decoder.decode(value, { stream: true });
    
    // Split by the SSE message separator
    let parts = responseBuffer.split('\n\n');
    
    // Keep the last part in the buffer (it might be incomplete)
    responseBuffer = parts.pop();

    for (const part of parts) {
        const line = part.trim();
        if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') continue;
            try {
                const data = JSON.parse(dataStr);
                // Now safely update UI
            } catch (e) {
                console.error("Malformed frame:", line);
            }
                    }
                }
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                contentSpan.innerHTML += `<p style="color: red">Error: ${err.message}</p>`;
            }
        } finally {
            this.finalizeStream(fullText);
        }
    }

    renderIncremental(container, text) {
        // Optimization: Pre-process text to handle unfinished code blocks
        let processedText = text;
        const openCodeBlocks = (text.match(/```/g) || []).length;
        if (openCodeBlocks % 2 !== 0) {
            processedText += '\n```'; // Temporarily close for rendering
        }

        // Render Markdown
        const htmlContent = marked.parse(processedText);
        
        // Efficient DOM update: Only update if changed
        if (container.dataset.lastHash !== htmlContent.length) {
            container.innerHTML = htmlContent;
            container.dataset.lastHash = htmlContent.length;
            
            // Post-process Mermaid blocks
            this.processMermaidBlocks(container);
        }
        
        this.scrollToBottom();
    }

    processMermaidBlocks(container) {
        const codeBlocks = container.querySelectorAll('pre code.language-mermaid');
        codeBlocks.forEach((block, index) => {
            const parent = block.parentElement;
            const code = block.textContent.trim();
            
            // Check if Mermaid code is complete
            if (this.isMermaidComplete(code)) {
                const containerId = `mermaid-container-${index}`;
                let mermaidDiv = parent.nextElementSibling;
                
                if (!mermaidDiv || !mermaidDiv.classList.contains('mermaid-container')) {
                    mermaidDiv = document.createElement('div');
                    mermaidDiv.id = containerId;
                    mermaidDiv.className = 'mermaid-container';
                    parent.after(mermaidDiv);
                    parent.style.display = 'none'; // Hide raw code
                }
                
                // Debounce Mermaid rendering
                this.mermaidManager.renderDiagram(containerId, code);
            }
        });
    }

    isMermaidComplete(code) {
        // Basic heuristic: check if it has common Mermaid starting keywords and minimum lines
        const starts = ['graph', 'sequenceDiagram', 'gannt', 'classDiagram', 'erDiagram', 'stateDiagram', 'pie', 'flowchart'];
        const hasStart = starts.some(s => code.trim().startsWith(s));
        return hasStart && code.split('\n').length > 1;
    }

    finalizeStream(finalText) {
        if (finalText) {
            this.messages.push({ role: 'assistant', content: finalText });
        }
        this.toggleLoading(false);
        this.abortController = null;
    }

    toggleLoading(isLoading) {
        this.sendBtn.classList.toggle('hidden', isLoading);
        this.stopBtn.classList.toggle('hidden', !isLoading);
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    window.engine = new StreamingEngine();
});

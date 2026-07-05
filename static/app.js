/**
 * Groq Mirror Professional - Core Frontend Engine
 * 
 * Features:
 * - Persistent Line Buffer (Handles fragmented TCP packets)
 * - Throttled Render Scheduler (requestAnimationFrame at 60fps)
 * - Atomic Mermaid Rendering (Prevents re-render flicker)
 * - Incremental Markdown (Auto-closes open tags during streaming)
 */

class MermaidManager {
    constructor() {
        this.renderCache = new Map(); // Cache to prevent redundant renders
        this.counter = 0;
        
        // Initialize Mermaid with dark theme
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            fontFamily: 'monospace'
        });
    }

    /**
     * Renders a Mermaid diagram only if the code has changed.
     */
    async renderDiagram(containerId, code) {
        // Skip if this specific container already rendered this exact code
        if (this.renderCache.get(containerId) === code) return;

        const container = document.getElementById(containerId);
        if (!container) return;

        try {
            const uniqueId = `mermaid-svg-${Date.now()}-${this.counter++}`;
            // Use mermaid.render (Async API) instead of init()
            const { svg } = await mermaid.render(uniqueId, code);
            
            container.innerHTML = svg;
            this.renderCache.set(containerId, code);
            
            // Fix SVG sizing
            const svgElement = container.querySelector('svg');
            if (svgElement) {
                svgElement.style.maxWidth = '100%';
                svgElement.style.height = 'auto';
            }
        } catch (err) {
            console.error("Mermaid Syntax Error:", err);
            // Don't show error immediately during streaming as code might be partial
        }
    }
}

class StreamingEngine {
    constructor() {
        // DOM Elements
        this.chatContainer = document.getElementById('chat-container');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        
        // Internal State
        this.mermaidManager = new MermaidManager();
        this.abortController = null;
        this.messages = [];
        
        // Rendering State (The Source of Truth)
        this.streamingState = {
            fullText: "",
            isDirty: false,
            container: null,
            lastRenderedHtml: ""
        };

        this.initMarked();
        this.initEventListeners();
        this.startAnimationLoop();
    }

    initMarked() {
        marked.setOptions({
            highlight: (code, lang) => {
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

        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = (this.userInput.scrollHeight) + 'px';
        });
    }

    /**
     * High-performance Loop
     * Synchronizes DOM updates with the browser's refresh rate.
     */
    startAnimationLoop() {
        const frame = () => {
            if (this.streamingState.isDirty && this.streamingState.container) {
                this.renderIncremental();
                this.streamingState.isDirty = false;
            }
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }

    async handleSend() {
        const text = this.userInput.value.trim();
        if (!text) return;

        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        
        // Add User Message
        this.addMessage(text, 'user');
        this.messages.push({ role: 'user', content: text });

        await this.startStreaming();
    }

    async startStreaming() {
        this.toggleUIState(true);
        this.abortController = new AbortController();
        
        const botMsgDiv = this.addMessage('', 'bot');
        const contentArea = document.createElement('div');
        botMsgDiv.appendChild(contentArea);

        // Reset state for new stream
        this.streamingState.fullText = "";
        this.streamingState.container = contentArea;
        this.streamingState.lastRenderedHtml = "";
        this.streamingState.isDirty = false;

        let lineBuffer = ""; // Crucial: Accumulates partial JSON chunks

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

                lineBuffer += decoder.decode(value, { stream: true });
                
                // Process only complete lines
                let lines = lineBuffer.split('\n');
                lineBuffer = lines.pop(); // Keep partial line for next chunk

                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
                    
                    const dataStr = trimmedLine.slice(6);
                    if (dataStr === '[DONE]') continue;

                    try {
                        const data = JSON.parse(dataStr);
                        if (data.content) {
                            this.streamingState.fullText += data.content;
                            this.streamingState.isDirty = true;
                        }
                    } catch (e) {
                        // If JSON fails, it's a protocol fragmentation error
                        // Prepend back to buffer
                        lineBuffer = line + '\n' + lineBuffer;
                    }
                }
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                contentArea.innerHTML += `<div class="error">Stream Error: ${err.message}</div>`;
            }
        } finally {
            this.finalizeStream();
        }
    }

    renderIncremental() {
        let textToParse = this.streamingState.fullText;

        // 1. Fix Markdown integrity (close open code blocks)
        const codeBlockOccurrences = (textToParse.match(/```/g) || []).length;
        if (codeBlockOccurrences % 2 !== 0) {
            textToParse += '\n```';
        }

        // 2. Generate HTML
        const htmlOutput = marked.parse(textToParse);

        // 3. Diff check (Prevent unnecessary DOM churn)
        if (this.streamingState.lastRenderedHtml !== htmlOutput) {
            this.streamingState.container.innerHTML = htmlOutput;
            this.streamingState.lastRenderedHtml = htmlOutput;
            
            // 4. Process Diagrams
            this.processDiagrams(this.streamingState.container);
            
            // 5. Sync Scroll
            this.scrollToBottom();
        }
    }

    processDiagrams(container) {
        const mermaidBlocks = container.querySelectorAll('pre code.language-mermaid');
        mermaidBlocks.forEach((block, index) => {
            const code = block.textContent.trim();
            
            // Check if diagram code is even minimally valid
            if (code.length < 10) return; 

            const parent = block.parentElement;
            const containerId = `mermaid-output-${index}`;
            
            let outputDiv = parent.nextElementSibling;
            if (!outputDiv || !outputDiv.classList.contains('mermaid-container')) {
                outputDiv = document.createElement('div');
                outputDiv.className = 'mermaid-container';
                outputDiv.id = containerId;
                parent.after(outputDiv);
                parent.style.display = 'none'; // Hide raw markdown code
            }

            // Attempt render
            this.mermaidManager.renderDiagram(outputDiv.id, code);
        });
    }

    addMessage(text, role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message markdown-body`;
        
        if (role === 'user') {
            msgDiv.textContent = text;
        }
        
        this.chatContainer.appendChild(msgDiv);
        this.scrollToBottom();
        return msgDiv;
    }

    finalizeStream() {
        if (this.streamingState.fullText) {
            this.messages.push({ role: 'assistant', content: this.streamingState.fullText });
        }
        this.toggleUIState(false);
        this.abortController = null;
    }

    toggleUIState(isLoading) {
        this.sendBtn.classList.toggle('hidden', isLoading);
        this.stopBtn.classList.toggle('hidden', !isLoading);
    }

    handleStop() {
        if (this.abortController) this.abortController.abort();
    }

    clearChat() {
        this.chatContainer.innerHTML = '';
        this.messages = [];
    }

    scrollToBottom() {
        this.chatContainer.scrollTo({
            top: this.chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Global Initialization
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StreamingEngine();
});

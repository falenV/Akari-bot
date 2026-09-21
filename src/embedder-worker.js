// local MiniLM embedder runs in a dedicated worker thread so the CPU bound matrix math involved in generating embeddings never blocks the main thread's event loop (and therefore never risks delaying Discord.js's gateway heartbeat), Must live alongside bot.js, it's loaded via `new Worker(path.join(__dirname, 'embedder-worker.js'))`

import { parentPort } from 'node:worker_threads';

let embedder = null;

async function initEmbedder() {
    try {
        const { pipeline } = await import('@huggingface/transformers');
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        parentPort.postMessage({ type: 'ready' });
    } catch (err) {
        parentPort.postMessage({ type: 'init_error', error: err.message });
    }
}

parentPort.on('message', async (msg) => {
    if (msg.type !== 'embed') return;

    if (!embedder) {
        parentPort.postMessage({ type: 'result', id: msg.id, error: 'Embedder not initialized' });
        return;
    }

    try {
        const output = await embedder(msg.text, { pooling: 'mean', normalize: true });
        parentPort.postMessage({ type: 'result', id: msg.id, vector: Array.from(output.data) });
    } catch (err) {
        parentPort.postMessage({ type: 'result', id: msg.id, error: err.message });
    }
});

initEmbedder();

/**
 * pcm-processor.js — AudioWorklet processor for raw PCM16 capture
 *
 * Runs in the audio rendering thread. Converts float32 samples
 * from the microphone into int16 (the format Groq Whisper expects)
 * and posts them to the main thread via MessagePort.
 */

class PCMProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0] && input[0].length > 0) {
            const float32 = input[0];
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
                const s = Math.max(-1, Math.min(1, float32[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            this.port.postMessage(int16.buffer, [int16.buffer]);
        }
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);

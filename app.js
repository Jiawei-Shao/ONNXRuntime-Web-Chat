import { LLM } from './onnx.js'

let llm = undefined;

async function submitInput(e) {
    const input = document.getElementById('inputBox').value;
    const outputBox = document.getElementById('outputBox');
    outputBox.value = '';

    await llm.query(input, (word) => {
        const outputBox = document.getElementById('outputBox');
        outputBox.value = word;
    });
}

async function hasWebGPU() {
    if (!("gpu" in navigator)) {
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch(e) {
        return false;
    }
}

window.onload = () => {
    hasWebGPU().then((supported) => {
        if (!supported) {
            const outputBox = document.getElementById('outputBox');
            outputBox.value = 'Your GPU or browser does not support WebGPU shader-f16.';
            return;
        }

        llm = new LLM();
        
        llm.init().then(() => {
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.addEventListener('click', submitInput);
            const inputBox = document.getElementById('inputBox');
            inputBox.focus();
        });
    });
}

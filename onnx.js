import ort from 'onnxruntime-web/webgpu'
import { env, AutoTokenizer } from '@xenova/transformers';

const kUseLocalModel = true;

ort.env.wasm.wasmPaths = 'dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

env.localModelPath = 'model/Phi3.5';
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = 'dist/';

const kConfigFileName = 'config.json';
const kModelDataPath = 'onnx';
const kModelFileName = 'model_q4f16.onnx';
const kModelExternalDataFileName = 'model_q4f16.onnx_data';

let kConfigFileAbsolutePath = `${env.localModelPath}/${kConfigFileName}`;
let kModelFileAbsolutePath = `${env.localModelPath}/${kModelDataPath}/${kModelFileName}`;
let kModelExternalDataAbsolutePath = `${env.localModelPath}/${kModelDataPath}/${kModelExternalDataFileName}`;

if (!kUseLocalModel) {
    kConfigFileAbsolutePath = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main";
    kModelFileAbsolutePath = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/onnx/model_q4f16.onnx";
    kModelExternalDataAbsolutePath = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/onnx/model_q4f16.onnx_data";
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
}

const kOfficialPhi3ONNXModelRepo = 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web';

const kMaxOutputTokens = 4096;

class Tokenizer {
    tokenizer = undefined;
    async init() {
        this.tokenizer = await AutoTokenizer.from_pretrained('./');
    }
    async TokenizePrompt(prompt) {
        const promptTokenizerResult = await this.tokenizer(
            prompt, { return_tensor: false, padding: true, truncation: true });
        const promptTokens = promptTokenizerResult.input_ids;
        return new ort.Tensor(
            'int64', BigInt64Array.from(promptTokens.map(BigInt)),
            [1, promptTokens.length]);
    }

    TokensToText(tokens, startidx) {
        return this.tokenizer.decode(
            tokens.slice(startidx), { skip_special_tokens: false, });
    }
}

export class LLM {
    tokenizer = undefined;
    inferenceSession = undefined;

    kv_dims = [];
    num_hidden_layers = 0;

    eos = 0n;

    constructor() {}

    // ...

    async init() {
        this.tokenizer = new Tokenizer();
        await this.tokenizer.init();

        const modelBytes =
            await this.fetchAndCache(kModelFileAbsolutePath);
        const modelExternalData =
            await this.fetchAndCache(kModelExternalDataAbsolutePath);
        let modelSize = modelBytes.byteLength + modelExternalData.byteLength;
        console.log(`${Math.round(modelSize / 1024 / 1024)} MB`);

        const jsonBytes = await this.fetchAndCache(kConfigFileAbsolutePath);
        const textDecoder = new TextDecoder();
        const modelConfig = JSON.parse(textDecoder.decode(jsonBytes));
        const inferenceSessionOptions = {
            executionProviders: ["webgpu"],
            preferredOutputLocation: {},
            externalData: [
                {
                    data: modelExternalData,
                    path: kModelExternalDataFileName,
                },
            ],
        };
        for (let i = 0; i < modelConfig.num_hidden_layers; ++i) {
            inferenceSessionOptions.preferredOutputLocation
                [`present.${i}.key`] = 'gpu-buffer';
            inferenceSessionOptions.preferredOutputLocation
                [`present.${i}.value`] = 'gpu-buffer';
        }
        this.inferenceSession =
            await ort.InferenceSession.create(modelBytes, inferenceSessionOptions);
        console.log('Create session success!');

        this.eos = modelConfig.eos_token_id;
        this.num_hidden_layers = modelConfig.num_hidden_layers;
        this.kv_dims =
            [1, modelConfig.num_key_value_heads, 0,
             modelConfig.hidden_size / modelConfig.num_attention_heads];
    }

    async query(input, callback) {
        let prompt =
            `<|system|>
            You are a friendly assistant.<|end|>
            <|user|>
            ${input}<|end|>
            <|assistant|>`;
        const inferenceInputIds = await this.tokenizer.TokenizePrompt(prompt);

        let feed = {};
        const empty = new Uint16Array();
        for (let i = 0; i < this.num_hidden_layers; ++i) {
            feed[`past_key_values.${i}.key`] = new ort.Tensor('float16', empty, this.kv_dims);
            feed[`past_key_values.${i}.value`] = new ort.Tensor('float16', empty, this.kv_dims);
        }

        feed['input_ids'] = inferenceInputIds;
        const output_tokens = [];
        output_tokens.push(...inferenceInputIds.data);

        let seqlen = output_tokens.length;
        const input_len = inferenceInputIds.size;
        feed['position_ids'] = new ort.Tensor(
            'int64', BigInt64Array.from({ length: input_len },
                (_, i) => BigInt(seqlen - input_len + i)),
                [1, input_len]);

        console.log('Start inferencing.')
        const promptTokensCount = inferenceInputIds.size;
        let last_token = 0n;
        // 32007 is |<end>| according to tokenizer.js so it is also an ending.
        while (last_token != this.eos && last_token != 32007 && seqlen < kMaxOutputTokens) {
            
            seqlen = output_tokens.length;
            feed['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);

            const outputs = await this.inferenceSession.run(feed);
            last_token = BigInt(this.argmax(outputs.logits));
            output_tokens.push(last_token);

            const text = this.tokenizer.TokensToText(output_tokens, promptTokensCount);
            callback(text);
        
            this.update_kv_cache(outputs, feed);
            feed['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([last_token]), [1, 1]);
            feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqlen)]), [1, 1]);
        }
        console.log('Inferencing completed!')
    }

    argmax(t) {
        const arr = t.data;
        const start = t.dims[2] * (t.dims[1] - 1);
        let max = arr[start];
        let maxidx = 0;
    
        for (let i = 0; i < t.dims[2]; i++) {
            const val = arr[i + start];
            if (!isFinite(val)) {
                throw new Error("found infinitive in logits");
            }
            if (val > max) {
                max = arr[i + start];
                maxidx = i;
            }
        }
        return maxidx;
    }

    update_kv_cache(outputs, feed) {
        for (const name in outputs) {
            if (name.startsWith('present')) {
                let newName = name.replace('present', 'past_key_values');
                const t = feed[newName];
                if (t.location === 'gpu-buffer') {
                    t.dispose();
                }
                feed[newName] = outputs[name];
            }
        }
    }

    async fetchAndCache(url) {
        try {
            const cache = await caches.open("onnx");
            let cachedResponse = await cache.match(url);
            if (cachedResponse === undefined) {
                console.log(`${url} (network)`);
                const buffer = await fetch(url).then(response => response.arrayBuffer());
                try {
                    await cache.put(url, new Response(buffer));
                } catch (error) {
                    console.error(error);
                    console.log(`The model should be downloaded from ${kOfficialPhi3ONNXModelRepo} and put into ${env.localModelPath}`);
                }
                return buffer;
            }
            console.log(`${url} (cached)`);
            const data = await cachedResponse.arrayBuffer();
            return data;
        } catch (error) {
            console.log(`can't fetch ${url}`);
            throw error;
        }
    }
}

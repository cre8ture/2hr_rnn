
document.getElementById('predictButton').disabled = true;
document.getElementById('predictButton').innerText = 'Please wait model is training....';
// Compile the model
const model = tf.sequential();
model.add(tf.layers.simpleRNN({
    units: 20,
    recurrentInitializer: 'GlorotNormal',
    inputShape: [1, 1]
}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Sample text to train the model
const text = `The TF-IDF (Term Frequency-Inverse Document Frequency) score is not a singular score for each document. Instead, it's a **vector** where each element represents a unique word in the document. 

Here's how it works:
- The **TF (Term Frequency)** part measures how frequently a term occurs in a document. If a word occurs often in a document, then the TF score for that word will be high.
- The **IDF (Inverse Document Frequency)** part measures the importance of a term in the entire corpus (a collection of documents). If a word occurs in many documents, it's not a unique identifier, so the IDF score for that word will be low.

By multiplying these two parts together, we get the TF-IDF score for each word in each document. This results in a vector for each document, where each element in the vector corresponds to the TF-IDF score for a different word. 

So, in summary, each document doesn't get a singular TF-IDF score. Instead, it gets a vector of TF-IDF scores, one for each unique word in the document. This vector can then be used in various applications like information retrieval, text mining, and user modeling.`;
// Create a vocabulary of unique words
const words = text.split(' ');
const vocab = Array.from(new Set(words));
const word2index = {};
const index2word = {};
vocab.forEach((word, i) => {
    word2index[word] = i;
    index2word[i] = word;
});

// // Convert the text to a tensor
// const textTensor = tf.tensor(text.split('').map(char => char.charCodeAt(0))).reshape([-1, 1, 1]);
// const targetTensor = tf.tensor(text.split('').map(char => char.charCodeAt(0))).reshape([-1, 1]);
const textTensor = tf.tensor(words.map(word => word2index[word])).reshape([-1, 1, 1]);
const targetTensor = tf.tensor(words.map(word => word2index[word])).reshape([-1, 1]);


// Train the model

// Train the model
model.fit(textTensor, targetTensor, {epochs: 10}).then(() => {
    // Enable the prediction button after the model is trained
    document.getElementById('predictButton').disabled = false;
    document.getElementById('predictButton').innerText = 'Predict Next Word RNN [TensorFlow]';
});
// model.fit(textTensor, targetTensor, {epochs: 10}).then(() => {
//     // Enable the prediction button after the model is trained
//     document.getElementById('predictButton').disabled = false;
//     document.getElementById('predictButton').innerText = 'Predict Next Letter RNN [TensorFlow]';
// });



// Event listener for the prediction button
document.getElementById('predictButton').addEventListener('click', function() {
    const inputText = document.getElementById('textInput').value;
    if (model && inputText) {
        const prediction = model.predict(tf.tensor([word2index[inputText]]).reshape([1, 1, 1]));
        const predictedWordIndex = prediction.argMax(-1).dataSync()[0];
        const predictedWord = index2word[predictedWordIndex];
        console.log('Predicted word:', predictedWord);
        console.log('Vector:', prediction.arraySync());
        document.getElementById('result').innerText = 'Predicted word: ' + predictedWord + '\nVector: ' + prediction.arraySync();
document.getElementById('result').style.overflowX = 'auto';
document.getElementById('result').style.whiteSpace = 'wrap';
document.getElementById('result').style.maxWidth = '400px';

    }
});

class RNN {
    constructor(inputSize, outputSize, hiddenSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenSize = hiddenSize;

        this.weights_ih = new Array(hiddenSize).fill().map(() => new Array(inputSize).fill(Math.random()));
        this.weights_ho = new Array(outputSize).fill().map(() => new Array(hiddenSize).fill(Math.random()));
        this.weights_hh = new Array(hiddenSize).fill().map(() => new Array(hiddenSize).fill(Math.random()));
    }

    tanh(x) {
        return Math.tanh(x);
    }

    forward(input, hidden) {
        const newHidden = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                newHidden[i] += this.weights_ih[i][j] * input[j];
            }
            for (let j = 0; j < this.hiddenSize; j++) {
                newHidden[i] += this.weights_hh[i][j] * hidden[j];
            }
            newHidden[i] = this.tanh(newHidden[i]);
        }

        const output = new Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                output[i] += this.weights_ho[i][j] * newHidden[j];
            }
            output[i] = this.tanh(output[i]);
        }

        return { output, newHidden };
    }
}

document.getElementById('predictButtonNaive').addEventListener('click', function() {
    const inputText = document.getElementById('textInput').value;
    const outputSize = inputText.length;
    const hiddenSize = inputText.length;
    const rnn = new RNN(inputText.length, outputSize, hiddenSize);

    if (rnn && inputText) {
        // Predict with self-built model
        const hidden = new Array(hiddenSize).fill(0);
        const input = inputText.split('').map(char => char.charCodeAt(0));
        const { output } = rnn.forward(input, hidden);
        const predictedChar = String.fromCharCode(Math.round(output[0]));
        console.log("output, predictedChar",output, predictedChar);

        document.getElementById('resultSelfBuilt').innerText = 'Predicted letter (Self-built): ' + predictedChar + '\nVector: ' + JSON.stringify(output);
        document.getElementById('resultSelfBuilt').style.maxWidth = '400px';
document.getElementById('resultSelfBuilt').style.overflowX = 'auto';
document.getElementById('resultSelfBuilt').style.whiteSpace = 'wrap';
    }
});
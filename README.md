# [TF-KMeans](https://github.com/pratikpc/TF-KMeans)

## Description

A Simple JavaScript Library to make it easy for people to use KMeans algorithms with Tensorflow JS.

The library was born out of another project in which except KMeans, our code completely depended on TF.JS

As such, moving to TF.JS helped standardise our code base substantially and reduce dependency on other libraries

## [Sample Code](./index.js)

When you are using a browser at frontend!

Example takes cosineDistance.

~~~javascript
const KMeans = require('./tf-kmeans').default
const tf = require('@tensorflow/tfjs')

function testCosineCluster() {
  tf.tidy(() => {
    const kmeans = new KMeans({
      k: 3,
      maxIter: 50,
      distanceFunction: KMeans.cosineDistance,
    })
    const data1 = [
      [1, 23, 3],
      [1, 23, 3],
      [4, 5, 2.1],
      [2, 3, 1],
      [4, 5, 2],
      [4, 5, 2],
      [4, 5, 2],
      [4, 5, 2],
      [4, 5, 2],
      [4, 5, 2.1],
      [4, 5, 2.1],
      [4, 5, 2.1],
      [4, 5, 2.1],
    ]
    const data2 = [
      [-0.026, 0.0533, 0.1],
      [0.1, 0.033, 0.032],
      [0.12, -0.2, 0.123],
      [0.333333, 0.21, 0.21],
      [-0.76, -0.321, 0.228],
      [-0.26, -0.321, 0.22],
      [0.1, 0.3, 0.28],
      [0.1, 0.06, 0.22],
      [-0.00000001, 0.01, 0.0211],
      [0.02, -0.009, -0.0211],
      [0.12, 0.01, 0.0211],
      [0.02, 0.01, -0.111],
      [-0.02333, -0.043, -0.12001],
    ]
    const dataset = tf.tensor(data2)
    const train = kmeans.train(dataset)

    console.log('Train Classify', train.arraySync())
    console.log('Centers', kmeans.centroids.arraySync())
    console.log('Memory Used', tf.memory())

    console.log('Predict:')
    const ys = kmeans.predict(
      tf.tensor([
        [0.1, 0.22, 0.21],
        [-0.02, -0.01, 0.02001],
      ]),
    )

    console.log('--------category index--------')
    console.log(ys.index.arraySync())
    console.log('--------category center-------')
    console.log(ys.center.arraySync())
    console.log('--------category ditance-------')
    console.log(ys.distance.arraySync())

    // dispose
    kmeans.dispose()
    train.dispose()
    dataset.dispose()
  })
}

testCosineCluster()

~~~

When you are using nodejs at backend!

Example takes euclideanDistance.

~~~javascript
const KMeans = require('./tf-kmeans-node').default
const tf = require('@tensorflow/tfjs-node')
const PATH = './kmeans.json'

function test() {
  tf.tidy(() => {
    const kmeans = new KMeans({
      k: 3,
      maxIter: 50,
    })
    console.log(kmeans)
    const dataset = tf.tensor([
      [2, 2, 2],
      [5, 5, 5],
      [3, 3, 3],
      [4, 4, 4],
      [7, 8, 7],
    ])
    const train = kmeans.train(dataset)
    console.log('Train Classify', train.arraySync())
    console.log('Centers', kmeans.centroids.arraySync())
    console.log('Memory Used', tf.memory())

    console.log('Predict:')
    const pre = kmeans.predict(tf.tensor([2, 3, 2]))
    console.log('Category index:', pre.index.arraySync())
    console.log('Category distance:', pre.distance.arraySync())
    console.log('Category center:', pre.center.arraySync())

    kmeans.save(PATH)

    // dispose
    kmeans.dispose()
    train.dispose()
    dataset.dispose()
  })
}
function testLoad() {
  console.log('====================Test load model=======================')
  const model = require(PATH)
  const kmeans = new KMeans(model)
  console.log('Predict:')
  const pre = kmeans.predict(tf.tensor([2, 3, 2]))
  console.log('Category index:', pre.index.arraySync())
  console.log('Category distance:', pre.distance.arraySync())
  console.log('Category center:', pre.center.arraySync())
}

// train
test()
// load
testLoad()
~~~

## Functions

1. ***`Constructor`***

    Takes 4 Optional parameters
    - k:-                Number of Clusters
    - maxIter:-          Max Iterations
    - distanceFunction:- The Distance function Used Currently: `euclideanDistance` and `cosineDistance`
    - centroids?:-        Always when loading from a save json model, you don't need to train again.

2. ***`train`***

    Takes Dataset as Parameter

    Performs Training on This Dataset

    Sync callback function is *optional*

3. ***`trainAsync`***

    Takes Dataset as Parameter

    Performs Training on This Dataset

    Also takes *async* callback function called at the end of every iteration

5. ***`predict`***

    Performs Predictions on the data Provided as Input
    Output: `{ index: category index, distance: distance to category center, center: category center }`

6. ***`save`***

    Save trained k-means to a json file. Pls give a '/path/to/xxx.json' into it.

## PEER DEPENDENCIES

1. [TensorFlow.JS](https://www.tensorflow.org/js "tfjs")

## Typings

As the code is originally written in TypeScript, Type Support is provided out of the box

## Contact Me

You could contact me [devilyouwei](https://github.com/devilyouwei/tf-kmeans)

Thanks to [pratikpc](https://github.com/pratikpc/tf-kmeans)

You could file issues or add features via Pull Requests on GitHub

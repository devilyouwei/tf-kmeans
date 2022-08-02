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

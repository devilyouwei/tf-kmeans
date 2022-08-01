const KMeans = require('./tf-kmeans-node').default
const tf = require('@tensorflow/tfjs-node')

const PATH = './kmeans.json'

function test() {
  tf.tidy(() => {
    const kmeans = new KMeans({
      k: 3,
      maxIter: 10,
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
    console.log('Category index:')
    kmeans.predict(tf.tensor([2, 3, 2])).index.print()
    kmeans.predict(tf.tensor([5, 5, 4])).index.print()
    console.log('Category confidence:')
    kmeans.predict(tf.tensor([2, 3, 2])).confidence.print()
    kmeans.predict(tf.tensor([5, 5, 4])).confidence.print()

    kmeans.save(PATH)

    // dispose
    kmeans.dispose()
    train.dispose()
    dataset.dispose()
  })
}
function testLoad() {
  const model = require(PATH)
  const kmeans = new KMeans(model)
  console.log('Load Predict:')
  console.log('Category index:')
  kmeans.predict(tf.tensor([2, 3, 2])).index.print()
  kmeans.predict(tf.tensor([5, 5, 4])).index.print()
  console.log('Category confidence:')
  kmeans.predict(tf.tensor([2, 3, 2])).confidence.print()
  kmeans.predict(tf.tensor([5, 5, 4])).confidence.print()
}

// train
test()
// load
testLoad()

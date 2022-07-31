const KMeans = require('../dist/index').default
const tf = require('@tensorflow/tfjs')
const PATH = './kmeans.json'

function test() {
  const kmeans = new KMeans({
    k: 2,
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
  const predict = kmeans.train(dataset)
  console.log('Train Classify', predict.arraySync())
  console.log('Centers', kmeans.centroids.arraySync())
  console.log('Memory Used', tf.memory())
  console.log('Predict:')
  kmeans.predict(tf.tensor([2, 3, 2])).print()
  kmeans.predict(tf.tensor([5, 5, 4])).print()
  console.log('Save to', PATH)
  kmeans.save(PATH)
  kmeans.dispose()
  predict.dispose()
  dataset.dispose()
}

function loadTest() {
  const km = require(PATH)
  const kmeans = new KMeans(km)
  kmeans.predict(tf.tensor([2, 3, 2])).print()
  kmeans.predict(tf.tensor([5, 5, 4])).print()
}
test()
loadTest()

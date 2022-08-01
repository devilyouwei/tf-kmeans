const KMeans = require('./tf-kmeans').default
const tf = require('@tensorflow/tfjs')

function testCosineCluster() {
  tf.tidy(() => {
    const kmeans = new KMeans({
      k: 2,
      maxIter: 30,
      distanceFunction: KMeans.cosineDistance,
    })
    console.log(kmeans)
    const dataset = tf.tensor([
      [0.02, 0.033, 0.1],
      [0.1, 0.2, 0.1],
      [0.1, 0.2, 0.1],
      [0.3, 0.21, 0.21],
      [0.06, 0.321, 0.22],
      [0.1, 0.3, 0.22],
      [0.00000001, 0.01, 0.0211],
      [0.02, 0.009, 0.0211],
      [0.02, 0.01, 0.0211],
      [0.02, 0.01, 0.0211],
      [0.02, 0.01, 0.02001],
    ])
    const predict = kmeans.train(dataset)

    console.log('Train Classify', predict.arraySync())
    console.log('Centers', kmeans.centroids.arraySync())
    console.log('Memory Used', tf.memory())

    console.log('Predict:')
    const ys = kmeans.predict(
      tf.tensor([
        [0.1, 0.22, 0.21],
        [0.02, 0.01, 0.02001],
      ]),
    )
    console.log('--------category index--------')
    console.log(ys.index.arraySync())
    console.log('--------category center-------')
    ys.index.arraySync().forEach((v) => {
      console.log(kmeans.centroids.arraySync()[v])
    })
    console.log('--------category confidence-------')
    console.log(ys.confidence.arraySync())

    // dispose
    kmeans.dispose()
    predict.dispose()
    dataset.dispose()
  })
}

testCosineCluster()

const KMeans = require('./tf-kmeans').default
const tf = require('@tensorflow/tfjs')

function testCosineCluster() {
  tf.tidy(() => {
    const kmeans = new KMeans({
      k: 3,
      maxIter: 10,
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
      [-0.02, 0.033, 0.1],
      [0.1, 0.033, 0.02],
      [0.1, -0.2, 0.1],
      [0.3, 0.21, 0.21],
      [-0.06, -0.321, 0.22],
      [0.1, 0.3, 0.22],
      [-0.00000001, 0.01, 0.0211],
      [0.02, -0.009, -0.0211],
      [0.02, 0.01, 0.0211],
      [0.02, 0.01, -0.0211],
      [-0.02, -0.01, -0.02001],
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
    train.dispose()
    dataset.dispose()
  })
}

testCosineCluster()

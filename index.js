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

import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'

export default class KMeans {
    private k: number = 2
    private maxIter: number = 10
    private distanceFunction = KMeans.euclideanDistance
    private centroids!: tf.Tensor

    constructor({ k = 2, maxIter = 10, distanceFunction = KMeans.euclideanDistance, centroids = [] } = {}) {
        this.k = k
        this.maxIter = maxIter
        this.distanceFunction = distanceFunction

        if (centroids && centroids.length) {
            console.log('Recovering k-means model...')
            this.centroids = tf.tensor(centroids)
        }
    }

    public save(path: string) {
        const model = {
            k: this.k,
            maxIter: this.maxIter,
            centroids: this.centroids.arraySync()
        }
        // node use fs to save
        fs.writeFileSync(path, JSON.stringify(model))
        return model
    }

    public static euclideanDistance(values: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => values.squaredDifference(centroids).sum(1).sqrt())
    }

    private generateIndices(rows: number) {
        const indices: number[] = []
        indices.length = rows
        for (let i = 0; i < indices.length; ++i) indices[i] = i
        return indices
    }
    private newCentroidSingle(values: tf.Tensor, assignments: tf.Tensor, cluster: number, rows: number) {
        return tf.tidy(() => {
            // Make All Values Of Array to be of Same Size as Our Cluster
            let selectedIndices: number[] = []
            selectedIndices.length = rows
            selectedIndices = selectedIndices.fill(cluster)
            const selectedIndicesT = tf.tensor(selectedIndices)

            let where = tf.equal(assignments, selectedIndicesT).asType('int32')
            where = where.reshape([where.shape[0], 1])
            const count = where.sum()

            const newCentroid = values.mul(where).sum(0).div(count)
            return newCentroid
        })
    }
    private newCentroids(values: tf.Tensor, assignments: tf.Tensor) {
        return tf.tidy(() => {
            const rows = values.shape[0]
            const centroids: tf.Tensor[] = []
            for (let cluster = 0; cluster < this.k; ++cluster) {
                centroids.push(this.newCentroidSingle(values, assignments, cluster, rows))
            }
            return tf.stack(centroids)
        })
    }
    private assignCluster(value: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => this.distanceFunction(value, centroids).argMin(0))
    }
    private assignClusters(values: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => {
            const rows = values.shape[0]
            const minIndexes: tf.Tensor[] = []
            for (const index of this.generateIndices(rows)) {
                const value = values.gather(index)
                minIndexes.push(this.assignCluster(value, centroids))
                value.dispose()
            }
            return tf.stack(minIndexes)
        })
    }
    private randomSample(vals: tf.Tensor) {
        return tf.tidy(() => {
            const rows = vals.shape[0]
            if (rows < this.k) throw new Error('Rows are Less than K')

            const indicesRaw = tf.util.createShuffledIndices(rows).slice(0, this.k)
            const indices: number[] = []
            indicesRaw.forEach((index: number) => indices.push(index))
            // Extract Random Indices
            return tf.gatherND(vals, tf.tensor(indices, [this.k, 1], 'int32'))
        })
    }
    private checkCentroidSimmilarity(newCentroids: tf.Tensor, centroids: tf.Tensor, vals: tf.Tensor) {
        return tf.tidy(
            () =>
                newCentroids
                    .equal(centroids)
                    .asType('int32')
                    .sum(1)
                    .div(vals.shape[1]!)
                    .sum()
                    .equal(this.k)
                    .dataSync()[0]
        )
    }
    private trainSingleStep(values: tf.Tensor) {
        return tf.tidy(() => {
            const predictions = this.predict(values)
            const newCentroids = this.newCentroids(values, predictions)
            return [newCentroids, predictions]
        })
    }
    public train(values: tf.Tensor, callback = (_centroid: tf.Tensor, _predictions: tf.Tensor) => {}) {
        this.centroids = this.randomSample(values)
        let iter = 0
        while (true) {
            let [newCentroids, predictions] = this.trainSingleStep(values)
            const same = this.checkCentroidSimmilarity(newCentroids, this.centroids, values)
            if (same || iter >= this.maxIter) {
                newCentroids.dispose()
                return predictions
            }
            this.centroids.dispose()
            this.centroids = newCentroids
            ++iter
            callback(this.centroids, predictions)
        }
    }
    public async trainAsync(
        values: tf.Tensor,
        callback = async (_iter: number, _centroid: tf.Tensor, _predictions: tf.Tensor) => {}
    ) {
        this.centroids = this.randomSample(values)
        let iter = 0
        while (true) {
            let [newCentroids, predictions] = this.trainSingleStep(values)
            const same = this.checkCentroidSimmilarity(newCentroids, this.centroids, values)
            if (same || iter >= this.maxIter) {
                newCentroids.dispose()
                return predictions
            }
            this.centroids.dispose()
            this.centroids = newCentroids
            await callback(iter, this.centroids, predictions)
            ++iter
        }
    }
    public predict(y: tf.Tensor) {
        return tf.tidy(() => {
            if (y.shape[1] == null) y = y.reshape([1, y.shape[0]])
            return this.assignClusters(y, this.centroids)
        })
    }
    public dispose() {
        this.centroids.dispose()
    }
}
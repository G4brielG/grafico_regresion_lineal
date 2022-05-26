const modelo = tf.sequential();

modelo.add(tf.layers.dense({
  inputShape: 1,
  units: 1,
}));
modelo.add(tf.layers.dense({
  units: 3,
}))
modelo.add(tf.layers.dense({
  units: 1,
}))

modelo.compile({
  optimizer: "sgd",
  loss: "meanSquaredError",
});

const xs = tf.tensor([-1, 0, 1, 2, 3, 4], [6, 1])
const ys = tf.tensor([-1, 2, 5, 8, 11, 14], [6, 1]);

modelo
  .fit(xs, ys, { epochs: 500, })
  .then(() => {
    const TensorX = tf.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const datosTensor = modelo.predict(TensorX).dataSync();
    const [...valores] = datosTensor;
    // Separar los valores en x e Y
    const res = valores.map((y, x) => ({ x, y }));

    const series = ["y=3x+2"];
    const data = {
      values: [res],
      series
    };

    const surface = {
      name: "3x+2",
      tab: "Función"
    };

    tfvis.render.linechart(surface, data);
    tf.dispose([xs, ys, modelo, datosTensor, TensorX]);
  });

//Perdidas

const model = tf.sequential({
  layers: [
    tf.layers.dense({
      inputShape: 1,
      units: 1,
      activation: 'relu'
    })
  ]
});

model.compile({
  optimizer: 'sgd',
  loss: "meanSquaredError",
  metrics: ['accuracy']
});

const surface = {
  name: 'Pérdida',
  tab: 'Pérdida'
};

model.fit(xs, ys, {
  epochs: 10,
  batchSize: 32,
  callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
});
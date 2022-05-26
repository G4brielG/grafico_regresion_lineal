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
  metrics: ['accuracy']
});

const xs = tf.tensor([-1, 0, 1, 2, 3, 4], [6, 1])
const ys = tf.tensor([-1, 2, 5, 8, 11, 14], [6, 1]);

const surface0 = {
  name: 'Pérdida',
  tab: 'Pérdida'
};

modelo
  .fit(xs, ys, { epochs: 60, batchSize: 64, callbacks: tfvis.show.fitCallbacks(surface0, ['loss', 'acc']) })
  .then(() => {
    const TensorX = tf.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const datosTensor = modelo.predict(TensorX).dataSync();
    const [...valores] = datosTensor;
    // Separa los valores en x e Y
    const res = valores.map((y, x) => ({ x, y }));

    const series = ["y=3x+2"];
    const data = {
      values: [res],
      series
    };

    const surface1 = {
      name: "3x+2",
      tab: "Función"
    };

    tfvis.render.linechart(surface1, data);
    tf.dispose([xs, ys, modelo, datosTensor, TensorX]);
  });
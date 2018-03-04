module.exports = {
    mode: 'production',
    entry: './app.ts',
    output: {filename: 'bundle.js'},
    module: {rules: [{test: /\.ts$/, loaders: ['babel-loader', 'ts-loader']}]},
    resolve: {extensions: ['.js', '.ts']}
};

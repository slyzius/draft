using Iit.Research.ACE.BL.AccessLayers.ACE;
using Iit.Research.ACE.BL.Model;
using Iit.Research.ACE.BL.Services;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Iit.Research.ACE.BL.MachineLearning
{
    public class TrainingSample
    {
        public List<CharacteristicValue> Input { get; set; }
        public List<Aspect> Target { get; set; }
        public User User { get; set; }
    }

    /// <summary>
    /// Neural Network
    /// </summary>
    public class ANN
    {
        public List<Tuple<int, int>> _layers;
        public Matrix<double>[] _wHidden;
        public Vector<double>[] _wHiddenBias;
        public Matrix<double>[] _dwHidden;
        public Vector<double>[] _dwHiddenBias;

        public Vector<double> _lastLayerDerr;

        public List<Characteristic> _inputMap;
        public List<Aspect> _outputMap;
        int _numberOfHiddenFeatures = 32; //784;//32;

        double _learningRate = 0.001;
        double _momentumRate = 0;//0.01;

        public bool _testDerrivative = false;
        public bool _lastLinear = true;

        Random _rand = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode())).Value;

        #region ctor and initialization

        public ANN(List<Characteristic> inputMap, List<Aspect> outputMap)
        {
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();

            Initialize(inputMap, outputMap);
        }

        double Normal(double mean, double stdDev)
        {
            double u1 = _rand.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = _rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                        Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                        mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)

            return randNormal;
        }

        void Initialize(List<Characteristic> inputMap, List<Aspect> outputMap)
        {
            _inputMap = inputMap;
            _outputMap = outputMap;

            _layers = new List<Tuple<int, int>> { new Tuple<int, int>(_numberOfHiddenFeatures, inputMap.Count), new Tuple<int, int>(_numberOfHiddenFeatures, _numberOfHiddenFeatures), new Tuple<int, int>(outputMap.Count, /*outputMap.Count*/_numberOfHiddenFeatures) };

            _wHidden = _layers.Select(x => Matrix<double>.Build.Dense(x.Item1, x.Item2, Enumerable.Repeat<double>(0.0, x.Item1 * x.Item2)
                .Select(z =>
                {
                    var q = 1 / Math.Sqrt(x.Item2);
                    return Normal(0, q);

                }).ToArray())).ToArray();


            _wHiddenBias = _layers.Select(x => Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, x.Item1)
                .Select(z =>
                {
                    var q = 1 / Math.Sqrt(x.Item2);
                    return Normal(0, q);
                }).ToArray())).ToArray();


            _wHidden[_layers.Count - 1] = Matrix<double>.Build.Dense(_layers.Last().Item1, _layers.Last().Item2,
                Enumerable.Repeat<double>(0.0, _layers.Last().Item1 * _layers.Last().Item2).Select(z => _rand.NextDouble() / _layers.Last().Item1).ToArray());

            _wHiddenBias[_layers.Count - 1] = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, _layers.Last().Item1).Select(z => _rand.NextDouble() / _layers.Last().Item1).ToArray());
            _lastLayerDerr = Vector<double>.Build.Dense(Enumerable.Repeat<double>(1.0, _layers[_layers.Count - 1].Item1).ToArray());


            _dwHidden = _wHidden.Select(x => Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount, 0.0)).ToArray();
            _dwHiddenBias = _wHiddenBias.Select(x=> Vector<double>.Build.Dense(x.Count, 0.0)).ToArray();

            //TestDerrivatives();
        }

        #endregion

        #region activation func

        Vector<double> sigmoid(Vector<double> input, int layer)
        {
            if (_lastLinear && layer == _wHidden.Length - 1)
            {
                //return Vector<double>.Build.DenseOfArray(input.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray()); // sigmoid
                return input;
            }
            else
            {
                //return Vector<double>.Build.DenseOfArray(input.Select(x=> 1.0 /(1.0+Math.Exp(-x))).ToArray()); // sigmoid
                //return Vector<double>.Build.DenseOfArray(input.Select(x => Math.Max(0, x)).ToArray()); // reLu
                
                //return Vector<double>.Build.DenseOfArray(input.Select(x => 1.7159 * Math.Tanh(x * 2.0 / 3.0)).ToArray()); // tanh
                return (input * (2.0 / 3.0)).PointwiseTanh() * 1.7159;
                
                
                //return Vector<double>.Build.DenseOfArray(input.Select(x =>Math.Tanh(x)).ToArray()); // tanh
            }
        }

        Vector<double> derrivative(Vector<double> x, int layer)
        {
            if (_lastLinear && layer == _wHidden.Length)
            {
                //return x.PointwiseMultiply((1.0 - x)); //sigmoid
                return _lastLayerDerr;
            }
            else
            {
                //return x.PointwiseMultiply((1.0 - x)); //sigmoid
                //return Vector<double>.Build.DenseOfArray(x.Select(z => (z > 0) ? 1 : 0.00000 * z).ToArray()); // reLu
                
                return Vector<double>.Build.DenseOfArray(x.Select(z => (1.7159 - z * z / 1.7159) * 2.0 / 3.0 ).ToArray()); // tanh
                //return Vector<double>.Build.DenseOfArray(x.Select(z => (1- z * z) ).ToArray()); // tanh
            }
        }

        #endregion

        #region batch learning

        IEnumerable<TrainingSample> RandomValues(List<TrainingSample> traningSamples)
        {
            int size = traningSamples.Count;
            while (true)
            {
                yield return traningSamples[_rand.Next(size)];
            }
        }

        IEnumerable<List<TrainingSample>> RandomLists(IEnumerable<TrainingSample> values, int miniBatchSize, int degreeOfParallelism)
        {
            for(int i = 0; i < degreeOfParallelism; i++)
            {
                yield return values.Take(miniBatchSize / degreeOfParallelism).ToList();
            }
        }

        public double BatchUpdate(List<TrainingSample> traningSamples, int miniBatchSize = 64)
        {
            int mxDegreeOfParallelism = 4;

            double e = 0.0;
            var wHiddenDelta = _layers.Select(x => Matrix<double>.Build.Dense(x.Item1, x.Item2, 0.0)).ToArray();
            var wHiddenDeltaBias = _layers.Select(x => Vector<double>.Build.Dense(x.Item1, 0)).ToArray();

            object sync = new object();

            Parallel.ForEach<List<TrainingSample>>(RandomLists(RandomValues(traningSamples), miniBatchSize, mxDegreeOfParallelism), // source collection
                            new ParallelOptions { MaxDegreeOfParallelism = mxDegreeOfParallelism },

                            (trainingMiniBatchList) => // method invoked by the loop on each iteration
                            {
                                double el = 0.0;
                                var wHiddenDeltaLocal = wHiddenDelta.ToArray();
                                var wHiddenDeltaBiasLocal = wHiddenDeltaBias.ToArray();

                                foreach (var sample in trainingMiniBatchList)
                                {
                                    var outputs = MakeOutputs(MakeInputVector(sample.Input));
                                    var error = outputs.Last() - MakeOutputVector(sample.Target);

                                    el += Math.Pow(error.L2Norm(),2.0);

                                    // back propagation: derrivatives * error
                                    Matrix<double> w = null;
                                    for (int iLayer = _layers.Count - 1; iLayer >= 0; iLayer--)
                                    {
                                        Matrix<double> dWeigths = MakeLayerDerrivatives(w, iLayer, outputs, error);
                                        wHiddenDeltaBiasLocal[iLayer] += dWeigths.Column(dWeigths.ColumnCount - 1);
                                        wHiddenDeltaLocal[iLayer] += dWeigths.RemoveColumn(dWeigths.ColumnCount - 1);

                                        var dD = derrivative(outputs[iLayer + 1], iLayer + 1);
                                        var ww = _wHidden[iLayer].PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(dD, _layers[iLayer].Item2)));
                                        w = (w == null) ? ww : w * ww;
                                    
                                    }

                                }

                                lock (sync)
                                {
                                    e += el;

                                    for (int i = 0; i < _layers.Count; i++)
                                    {
                                        wHiddenDelta[i] += wHiddenDeltaLocal[i];
                                        wHiddenDeltaBias[i] += wHiddenDeltaBiasLocal[i];
                                    }
                                }
                            });
                

            // update network weights
            for (int i = 0; i < _layers.Count; i++)
            {
                /*if (_regularizationL2 > 0)
                {
                    _wHidden[i] -= wHiddenDelta[i] / miniBatchSize - _learningRate * _regularizationL2 * _wHidden[i];
                    _wHiddenBias[i] -= wHiddenDeltaBias[i] / miniBatchSize - _learningRate * _regularizationL2 * _wHiddenBias[i];
                }
                else*/
                {
                    _dwHidden[i] = _momentumRate * _dwHidden[i] - _learningRate * wHiddenDelta[i] / (miniBatchSize );
                    _dwHiddenBias[i] = _momentumRate * _dwHiddenBias[i] - _learningRate * wHiddenDeltaBias[i] / (miniBatchSize);
                    _wHidden[i] += _dwHidden[i];
                    _wHiddenBias[i] += _dwHiddenBias[i];
                }
            }
            return Math.Sqrt(e /= miniBatchSize);
        }

        public double OnlineUpdate(TrainingSample sample)
        {
            var outputs = MakeOutputs(MakeInputVector(sample.Input));
            var error = outputs.Last() - MakeOutputVector(sample.Target);

            //double se = error.Clone().PointwisePower(2).Sum();
            double se = Math.Pow(error.L2Norm(), 2.0);

            // back propagation: derrivatives * error
            Matrix<double> w = null;
            for (int iLayer = _layers.Count - 1; iLayer >= 0; iLayer--)
            {
                Matrix<double> dWeigths = MakeLayerDerrivatives(w, iLayer, outputs, error);
                _dwHiddenBias[iLayer] = _momentumRate * _dwHiddenBias[iLayer] - _learningRate * dWeigths.Column(dWeigths.ColumnCount - 1);
                _dwHidden[iLayer] = _momentumRate * _dwHidden[iLayer] - _learningRate * dWeigths.RemoveColumn(dWeigths.ColumnCount - 1);
                _wHidden[iLayer] += _dwHidden[iLayer];
                _wHiddenBias[iLayer] += _dwHiddenBias[iLayer];

                var dD = derrivative(outputs[iLayer + 1], iLayer + 1);
                var ww = _wHidden[iLayer].PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(dD, _layers[iLayer].Item2)));
                w = (w == null) ? ww : w * ww;

            }
            return se;
        }

        #endregion

        #region serialization

        public void WriteDB(IAceDbAccessLayer aceDbAccessLayer)
        {
            List<ANNHiddenFeaturesRecord> hiddenFeatures = new List<ANNHiddenFeaturesRecord>();
            List<ANNOutputAspectsRecord> outputAspects = new List<ANNOutputAspectsRecord>();

            // write hidden features layer
            for (int iHiddenFeature = 0; iHiddenFeature < _numberOfHiddenFeatures; iHiddenFeature++)
            {
                // weight for the feature bias
                hiddenFeatures.Add(new ANNHiddenFeaturesRecord { CharacteristicId = -1, ColumnId = _outputMap.Count, HiddenFeatureId = iHiddenFeature, Weight = _wHiddenBias[0][iHiddenFeature] });
      
                // store weights for the characteristics
                for (int iCol = 0; iCol < _inputMap.Count; iCol++)
                {
                    hiddenFeatures.Add(new ANNHiddenFeaturesRecord { CharacteristicId = _inputMap[iCol].Id, ColumnId = iCol, HiddenFeatureId = iHiddenFeature, Weight = _wHidden[0][iHiddenFeature, iCol] });
            
                }
            }

            // write output aspects layer weights
            for (int iAspect = 0; iAspect < _outputMap.Count; iAspect++)
            {
                // output weight bias
                outputAspects.Add(new ANNOutputAspectsRecord { AspectId = _outputMap[iAspect].Id, HiddenFeatureId = -1, Weight = _wHiddenBias[1][iAspect] });

                for (int iHiddenFeature = 0; iHiddenFeature < _numberOfHiddenFeatures; iHiddenFeature++)
                {
                    outputAspects.Add(new ANNOutputAspectsRecord { AspectId = _outputMap[iAspect].Id, HiddenFeatureId = iHiddenFeature, Weight = _wHidden[1][iAspect, iHiddenFeature] });
                }
            }

            aceDbAccessLayer.MergeANNHiddenFeatures(hiddenFeatures);
            aceDbAccessLayer.MergeANNOutputAspects(outputAspects);
        }

        public void ReadDB(IAceDbAccessLayer aceDbAccessLayer)
        {
            int iElem = 0;
            ANNHiddenFeaturesRecord[] hiddenFeatures = aceDbAccessLayer.GetANNHiddenFeatures(); 
            ANNOutputAspectsRecord[] outputAspects = aceDbAccessLayer.GetANNOutputAspects();

            // read hidden features layer
            for (int iHiddenFeature = 0; iHiddenFeature < _numberOfHiddenFeatures; iHiddenFeature++)
            {
                // weight for the feature bias
                _wHiddenBias[0][iHiddenFeature] = hiddenFeatures[iElem++].Weight;

                // store weights for the characteristics
                for (int iCol = 0; iCol < _inputMap.Count; iCol++)
                {
                    _wHidden[0][iHiddenFeature, iCol] = hiddenFeatures[iElem++].Weight;
                }
            }

            // read output aspects layer weights
            iElem = 0;
            for (int iAspect = 0; iAspect < _outputMap.Count; iAspect++)
            {
                // output weight bias
                _wHiddenBias[1][iAspect] = outputAspects[iElem++].Weight;

                for (int iHiddenFeature = 0; iHiddenFeature < _numberOfHiddenFeatures; iHiddenFeature++)
                {
                    _wHidden[1][iAspect, iHiddenFeature] = outputAspects[iElem++].Weight;
                }
                
            }
        }

        public void Write(TextWriter writer)
        {
            writer.WriteLine(_layers.Count);
            for(int i = 0; i < _layers.Count; i++)
            {
                writer.WriteLine(_layers[i].Item1);
                writer.WriteLine(_layers[i].Item2);

                WriteMatrix(writer, _wHidden[i], _layers[i].Item1, _layers[i].Item2);
                WriteVector(writer, _wHiddenBias[i]);
            }
            WriteVector(writer, _lastLayerDerr);
        }

        void WriteMatrix(TextWriter writer, Matrix<double> matrix, int rows, int cols)
        {
            var dense = matrix.Storage as DenseColumnMajorMatrixStorage<double>;
            if (dense != null)
            {
                foreach (var value in dense.Data)
                {
                    writer.WriteLine(value);
                }
            }
        }

        void WriteVector(TextWriter writer, Vector<double> vector)
        {
            var dense = vector.Storage as DenseVectorStorage<double>;
            if (dense != null)
            {
                foreach (var value in dense.Data)
                {
                    writer.WriteLine(value);
                }
            }
        }


        IEnumerable<double> ColumnMajor(TextReader reader, int total)
        {
            for (int i = 0; i < total; i++)
            {
                var value = double.Parse(reader.ReadLine());
                yield return value;
            }
        }

        Matrix<double> ReadMatrix(TextReader reader, int rows, int cols)
        {
            return Matrix<double>.Build.DenseOfColumnMajor(rows, cols, ColumnMajor(reader, rows*cols));
        }

        Vector<double> ReadVector(TextReader reader, int rows)
        {
            return Vector<double>.Build.Dense(ColumnMajor(reader, rows).ToArray());
        }

        public void Load(TextReader reader)
        {
            _layers = new List<Tuple<int, int>>();
            
            int NNlayers = int.Parse(reader.ReadLine()); ;
            _wHidden = new Matrix<double>[NNlayers];
            _wHiddenBias = new Vector<double>[NNlayers];

            for (int i = 0; i < NNlayers; i++)
            {
                int rows = int.Parse(reader.ReadLine());
                int cols = int.Parse(reader.ReadLine());
                _layers.Add(new Tuple<int, int>(rows,cols));

                _wHidden[i] = ReadMatrix(reader, rows, cols);
                _wHiddenBias[i] = ReadVector(reader, rows);
            }
            _lastLayerDerr = ReadVector(reader, _layers.Last().Item1);
        }

        #endregion

        #region Network

        public Dictionary<Aspect,double> Predict(List<CharacteristicValue> userCharacteristics)
        {
            var input = MakeInputVector(userCharacteristics);
            var outputs = MakeOutputs(input);
            var output = outputs.Last();

            int id = 0;
            return _outputMap.ToDictionary(x => x, x => output[id++]);
        }

        public Vector<double> PredictVector(List<CharacteristicValue> userCharacteristics)
        {
            var input = MakeInputVector(userCharacteristics);
            var outputs = MakeOutputs(input);
            var output = outputs.Last();
            return output;
        }

        Vector<double> MakeInputVector(List<CharacteristicValue> userCharacteristics)
        {
            var input = Vector<double>.Build.Dense(_inputMap.Count);
            var userCharacteristicsMap = userCharacteristics.ToDictionary(x => x.Characteristic, x => x.Score);

            for (int i = 0; i < _inputMap.Count; i++)
            {
                if (userCharacteristicsMap.ContainsKey(_inputMap[i]))
                {
                    input[i] = userCharacteristicsMap[_inputMap[i]];
                }
                else
                {
                    input[i] = 0.0;
                }
            }
            return input;
        }

        public Vector<double> MakeOutputVector(List<Aspect> aspects)
        {
            var output = Vector<double>.Build.Dense(_outputMap.Count);
            var aspectsMap = aspects.ToDictionary(x => x, x=> 1.0);

            for (int i = 0; i < _outputMap.Count; i++)
            {
                if (aspectsMap.ContainsKey(_outputMap[i]))
                {
                    output[i] = aspectsMap[_outputMap[i]];
                }
                else
                {
                    output[i] = 0.0;
                }
            }
            return output;
        }

        List<Vector<double>> MakeOutputs(Vector<double> input)
        {
            List<Vector<double>> out_net = new List<Vector<double>> { input };
            for (int h = 0; h < _wHidden.Length; h++)
            {
                var tnet_h = _wHidden[h] * out_net[h] + _wHiddenBias[h];
                out_net.Add(sigmoid(tnet_h, h));
            }
            return out_net;
        }

        Matrix<double> MakeLayerDerrivatives(Matrix<double> w, int layer, List<Vector<double>> outputs, Vector<double> error)
        {
            if (_testDerrivative)
            {
                int bpLayer1 = outputs.Count - 2;

                Matrix<double> w1 = null;
                for (; bpLayer1 > layer; bpLayer1--)
                {
                    var dD = derrivative(outputs[bpLayer1 + 1], bpLayer1 + 1);
                    var ww = _wHidden[bpLayer1].PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(dD, _layers[bpLayer1].Item2)));
                    w1 = (w1 == null) ? ww : w1 * ww;
                }

                if (layer != bpLayer1)
                {
                    throw new Exception("sss");
                }

                if (w != null)
                {
                    var wNorm = w.L2Norm();
                    var wNorm1 = w1.L2Norm();

                    if (wNorm != wNorm1)
                    {
                        throw new Exception("sss");
                    }
                }
            }

            // this is the layer we take a derrivative
            Matrix<double> ret = Matrix<double>.Build.Dense(_layers[layer].Item1, _layers[layer].Item2 + 1);
            var d1 = derrivative(outputs[layer + 1], layer + 1);
            var o1 = Vector<double>.Build.Dense(_layers[layer].Item1, 0.0);

            for (int row = 0; row < _layers[layer].Item1; ++row)
            {
                o1[row] = 1.0;
                var derrL = ((w != null) ? (w.Column(row) * error): o1 * error);

                for (int col = 0; col < _layers[layer].Item2; ++col)
                {
                    var do1 = d1[row] * outputs[layer][col];
                    //var derr = ((w != null) ? (w * o1) : o1) * error;
                    var derr = derrL * do1;

                    ret[row, col] = derr;

                    if (_testDerrivative)
                    {
                        var derrTestV = MakeNumericDerrivative(layer, row, col, false, outputs);
                        var derrTestW = derrTestV * error;
                        if (Math.Abs(derr - derrTestW) > 0.001)
                        {
                            throw new Exception("sss");
                        }
                    }
                }

                o1[row] = 0.0;
            }

            // and bias
            for (int row = 0; row < _layers[layer].Item1; ++row)
            {
                o1[row] = d1[row];
                //var dBias = (w != null) ? (w.Row(ioutput) * o1) : o1[ioutput];
                var dBias = ((w != null) ? (w * o1) : o1 ) * error;
                ret[row, _layers[layer].Item2] = dBias;

                if (_testDerrivative)
                {
                    var derrTestB = MakeNumericDerrivative(layer, row, 0, true, outputs) * error;
                    if (Math.Abs(dBias - derrTestB) > 0.001)
                    {
                        throw new Exception("The partial derrivative does not match numeric derrivative.");
                    }
                }
                o1[row] = 0.0;
            }

            return ret;
        }

        public Vector<double> MakeNumericDerrivative(int layer, int row, int col, bool bias, List<Vector<double>> outputs)
        {
            // numeric derrivative
            if (!bias)
            {
                double origin = _wHidden[layer][row, col];
                double alpha = 0.00000001;
                _wHidden[layer][row, col] = origin + alpha;
                var d1 = MakeOutputs(outputs[0]);
                _wHidden[layer][row, col] = origin - alpha;
                var d2 = MakeOutputs(outputs[0]);
                var derr = (d1.Last() - d2.Last()) / (2 * alpha);

                // restore weight
                _wHidden[layer][row, col] = origin;
                return derr;
            }
            else
            {
                double origin = _wHiddenBias[layer][row];
                double alpha = 0.00000001;
                _wHiddenBias[layer][row] = origin + alpha;
                var d1 = MakeOutputs(outputs[0]);
                _wHiddenBias[layer][row] = origin - alpha;
                var d2 = MakeOutputs(outputs[0]);
                var derr = (d1.Last() - d2.Last()) / (2 * alpha);

                // restore weight
                _wHiddenBias[layer][row] = origin;
                return derr;

            }
        }

        List<Vector<double>> Predict(Vector<double> input)
        {
            return MakeOutputs(input);
        }

        /*Tuple<Matrix<double>[], Vector<double>[]> LearnDelta(double error, int ioutput, List<Vector<double>> outputs)
        {

            int layer = 0;
            var wHiddenDelta = _layers.Select(x => Matrix<double>.Build.Dense(x.Item1, x.Item2, 0.0)).ToArray();
            var wHiddenDeltaBias = _layers.Select(x => Vector<double>.Build.Dense(x.Item1, 0)).ToArray();


            Func<double, double> round = (double d) =>
            {
                double dd = 1000000.0;
                return ((int)(d * 1000000.0)) / 1000000.0;
            };

            var dWeights = _layers.Select(x =>
            {
                var weights = Matrix<double>.Build.Dense(x.Item1, x.Item2, 0.0);
                var bias = Vector<double>.Build.Dense(x.Item1, _learningRate);

                Matrix<double> dWeigths = MakeLayerDerrivatives(layer, outputs, ioutput);

                for (int row = 0; row < x.Item1; row++)
                {
                    bias[row] = dWeigths[row, x.Item2];

                    if (_testDerrivative)
                    {
                        var derrTestB = MakeNumericDerrivative(layer, row, 0, true, outputs);
                        if (Math.Abs(round(dWeigths[row, x.Item2]) - round(derrTestB[ioutput])) > 0.0001)
                        {
                            throw new Exception("The partial derrivative does not match numeric derrivative.");
                        }
                    }

                    for (int col = 0; col < x.Item2; col++)
                    {
                            weights[row, col] = dWeigths[row, col];

                            // test derrivative
                            if (_testDerrivative)
                            {
                                var derrTestW = MakeNumericDerrivative(layer, row, col, false, outputs);
                                var d1 = round(dWeigths[row, col]);
                                var d2 = round(derrTestW[ioutput]);
                                if (Math.Abs(d1-d2) > 0.0001)
                                {
                                    throw new Exception("sss");
                                }
                            }
                    }
                }

                layer++;
                return new Tuple<Matrix<double>, Vector<double>>(weights, bias);
            }).ToArray();

            for (int i = 0; i < _layers.Count; i++)
            {
                wHiddenDelta[i] += error * dWeights[i].Item1;
                wHiddenDeltaBias[i] += error * dWeights[i].Item2;
            }
            return new Tuple<Matrix<double>[], Vector<double>[]>(wHiddenDelta, wHiddenDeltaBias);
        }*/

        #endregion

        #region Test

        public static ANN Test()
        {

            // define a model
            List<Characteristic> inputMap = new List<Characteristic>
            {
                new Characteristic(1),
                new Characteristic(2),
                new Characteristic(3),
                new Characteristic(4),
                new Characteristic(5)
            };

            List<Aspect> outputMap = new List<Aspect>
            {
                new Aspect(1,"eurusd",MetricGroup.Marketing),
                new Aspect(2,"french",MetricGroup.Marketing),
                new Aspect(3,"EU",MetricGroup.Marketing),
            };

            var nn = new ANN(inputMap, outputMap);
            return nn;
        }

        public void TestTrain()
        {
            _testDerrivative = true;

            // define a test data

            // items
            /*
            ai1<- c(1,1,0); # articles about eur are interesting for eurusd & french
            ai2 <- c(1,0,1); # articles about EU are interesting to all but not french
            ai3 <- c(0,1,1); # likes if it is about EUR or EU but not both together
            ai4 <- c(0,0,1);
            */
            List<Aspect> aiAll = new List<Aspect>
            {
                new Aspect(1,"1",MetricGroup.Marketing),
                new Aspect(2,"2",MetricGroup.Marketing),
                new Aspect(3,"3",MetricGroup.Marketing),
            };

            List<Aspect> ai1 = new List<Aspect>
            {
                new Aspect(1,"1",MetricGroup.Marketing),
                new Aspect(2,"2",MetricGroup.Marketing),
            };

            List<Aspect> ai2 = new List<Aspect>
            {
                new Aspect(1,"1",MetricGroup.Marketing),
                new Aspect(3,"3",MetricGroup.Marketing),
            };

            List<Aspect> ai3 = new List<Aspect>
            {
                new Aspect(2,"2",MetricGroup.Marketing),
                new Aspect(3,"3",MetricGroup.Marketing),
            };

            List<Aspect> ai4 = new List<Aspect>
            {
                new Aspect(3,"3",MetricGroup.Marketing),
            };

            // users
            /*
            ui1 <- c(1, 1 ,0,0,0);
            ui2 < -c(1, 0, 1, 0, 0);
            ui3 < -c(0, 0, 0, 1, 1);
            ui4 < -c(0, 0, 0, 0, 1);
            */
            List<CharacteristicValue> ui1 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(1),1.0,true),
                new CharacteristicValue(new Characteristic(2),1.0,true)
            };

            List<CharacteristicValue> ui2 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(1),1.0,true),
                new CharacteristicValue(new Characteristic(3),1.0,true),
            };

            List<CharacteristicValue> ui3 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(4),1.0,true),
                new CharacteristicValue(new Characteristic(5),1.0,true),
            };

            List<CharacteristicValue> ui4 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(5),1.0,true),
            };

            // preferences for training data userId x itemId eg. (2,3)  - user 2 likes/reads item 3
            var trainingData = new List<Tuple<List<CharacteristicValue>, List<Aspect>>>
                {
                    new Tuple<List<CharacteristicValue>,List<Aspect>>(ui1,ai1),
                    new Tuple<List<CharacteristicValue>,List<Aspect>>(ui1,ai2),
                    new Tuple<List<CharacteristicValue>,List<Aspect>>(ui2,ai2),
                    new Tuple<List<CharacteristicValue>,List<Aspect>>(ui2,ai3)
                };

            var rand = new Random();
            var sampleData = trainingData.SelectMany(x => Enumerable.Repeat(new TrainingSample { Input = x.Item1, Target = x.Item2 }, 250)).OrderByDescending(x => rand.NextDouble()).Take(2000).ToList();


            foreach (var sample in sampleData)
            {
                //var err = BatchUpdate(sampleData.Skip(i%2000).Take(1).ToList(), 1);
                var err = OnlineUpdate(sample);// BatchUpdate(sampleData, 4);

                //sse = OnlineUpdate(sampleData.Skip(i % 2000).First());

                /*if (Math.Abs(err - lastError) <= threshold)
                {
                    break;
                }*/
                //lastError = err;
                Debug.WriteLine(err);
            }
        }

        public Vector<double> PredictTest()
        {
            List<CharacteristicValue> ui1 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(1),1.0,true),
                new CharacteristicValue(new Characteristic(2),1.0,true)
            };

            List<CharacteristicValue> ui2 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(1),1.0,true),
                new CharacteristicValue(new Characteristic(3),1.0,true),
            };

            List<CharacteristicValue> ui3 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(4),1.0,true),
                new CharacteristicValue(new Characteristic(5),1.0,true),
            };

            List<CharacteristicValue> ui4 = new List<CharacteristicValue>
            {
                new CharacteristicValue(new Characteristic(5),1.0,true),
            };

            var u1 = MakeOutputs(MakeInputVector(ui1)).Last();
            var u2 = MakeOutputs(MakeInputVector(ui2)).Last();
            var u3 = MakeOutputs(MakeInputVector(ui3)).Last();

            return u1;

            
            /*var u1Predicted = aiAll.ToDictionary(x => x, x => u1[x.Id - 1]);
            var u2Predicted = aiAll.ToDictionary(x => x, x => u2[x.Id - 1]);
            var u3Predicted = aiAll.ToDictionary(x => x, x => u3[x.Id - 1]);

            var u1_a1 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u1Predicted, ai1),2);
            var u1_a2 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u1Predicted, ai2),2);
            var u1_a3 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u1Predicted, ai3),2);

            var u2_a1 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u2Predicted, ai1),2);
            var u2_a2 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u2Predicted, ai2),2);
            var u2_a3 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u2Predicted, ai3),2);

            var u3_a1 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u3Predicted, ai1),2);
            var u3_a2 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u3Predicted, ai2),2);
            var u3_a3 = Math.Round(SimilarityCalculator.CalculateCosine<Aspect>(u3Predicted, ai3),2);


            Debug.WriteLine(string.Format("U1: {0},{1},{2}",u1[0],u1[1],u1[2]));
            Debug.WriteLine(string.Format("U2: {0},{1},{2}", u2[0], u2[1], u2[2]));
            Debug.WriteLine(string.Format("U3: {0},{1},{2}", u3[0], u3[1], u3[2]));


            Debug.WriteLine(string.Format("U1x A1,A2,A3: {0},{1},{2}", u1_a1, u1_a2, u1_a3));
            Debug.WriteLine(string.Format("U2x A1,A2,A3: {0},{1},{2}", u2_a1, u2_a2, u2_a3));
            Debug.WriteLine(string.Format("U3x A1,A2,A3: {0},{1},{2}", u3_a1, u3_a2, u3_a3));
             */

        }


   /*     public void TestDerrivatives()
        {
            //testDerrivative = true;

            double r = 1.0;
            Random rand = new Random(123);
            var inputs = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, _layers[0].Item2).Select(z => ( (rand.NextDouble() > 0.5) ? 1.0 : 0)).ToArray());
            var targets = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, _layers[_layers.Count-1].Item1).Select(z => rand.NextDouble()).ToArray());

            int i = 0;
            while (true)
            {
                List<Vector<double>> outputs = MakeOutputs(inputs);
                Vector<double> p = outputs.Last();
                var err = p - targets;

                int ioputput = (int)((double)targets.Count * rand.NextDouble());

                Learn(err[ioputput], ioputput, outputs);
                if (i % 100 == 0)
                {
                    Debug.WriteLine(string.Format("{0}:{1}", i, err.Select(x=> x*x).Sum()));
                }
                i++;
            }

            //testDerrivative = false;
        }

        public void Learn(double error, int ioutput, List<Vector<double>> outputs)
        {
            var weights = LearnDelta(error, ioutput, outputs);
            for (int i = 0; i < _layers.Count; i++)
            {
                _wHidden[i] -= weights.Item1[i];
                _wHiddenBias[i] -= weights.Item2[i];
            }
        }*/

        #endregion

    }
}

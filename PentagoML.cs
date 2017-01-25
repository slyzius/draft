using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;


namespace Pentago
{
    public struct Winner
    {
        public List<int> _pos;
        public int _id;
    }

    public class Reward
    {
        List<Winner> _winningPos = new List<Winner> 
        {
            new Winner{ _pos = new List<int>{0,5,10,15}, _id = 1},
            new Winner{ _pos = new List<int>{3,6,9,12}, _id = 2},
            new Winner{ _pos = new List<int>{0,1,2,3}, _id = 3},
            new Winner{ _pos = new List<int>{4,5,6,7}, _id = 4},
            new Winner{ _pos = new List<int>{8,9,10,11}, _id = 5},
            new Winner{ _pos = new List<int>{12,13,14,15}, _id = 6},
            new Winner{ _pos = new List<int>{0,4,8,12}, _id = 7},
            new Winner{ _pos = new List<int>{1,5,9,13}, _id = 8},
            new Winner{ _pos = new List<int>{2,6,10,14}, _id = 9},
            new Winner{ _pos = new List<int>{3,7,11,15}, _id = 10},
        };

        Dictionary<int, List<int>> _winnerPos = new Dictionary<int, List<int>>();

        public Reward()
        {
            _winningPos.ForEach(x =>
            {
                x._pos.ForEach(y =>
                {
                    if (_winnerPos.ContainsKey(y))
                    {
                        _winnerPos[y].Add(x._id);
                    }
                    else
                    {
                        _winnerPos.Add(y, new List<int> { x._id });
                    }
                });
            });
        }

        public Tuple<float,float> GetReward(QSpace statePositions)
        {
            float r1 = 0, r2 = 0;

            // check if it is winning positions
            Dictionary<int, int> winningStatesMe = new Dictionary<int, int>();
            Dictionary<int, int> winningStatesOpponent = new Dictionary<int, int>();

            for (int i = 0; i < 16; ++i)
            {
                if (statePositions.IsMe(i))
                {
                    if (_winnerPos.ContainsKey(i))
                    {
                        _winnerPos[i].ForEach(x =>
                        {
                            if (winningStatesMe.ContainsKey(x))
                            {
                                winningStatesMe[x]++;
                            }
                            else
                            {
                                winningStatesMe.Add(x, 1);
                            }
                        });
                    }
                }
            }

            if (winningStatesMe.Where(x => x.Value == 4).Count() > 0)
            {
                r1 = 100;
            }

            for (int i = 0; i < 16; ++i)
            {
                if (statePositions.IsOpponent(i))
                {
                    if (_winnerPos.ContainsKey(i))
                    {
                        _winnerPos[i].ForEach(x =>
                        {
                            if (winningStatesOpponent.ContainsKey(x))
                            {
                                winningStatesOpponent[x]++;
                            }
                            else
                            {
                                winningStatesOpponent.Add(x, 1);
                            }
                        });
                    }
                }
            }

            if (winningStatesOpponent.Where(x => x.Value == 4).Count() > 0)
            {
                r2 = 100;
            }

            if (r1 > 0 || r2 > 0)
            {
                if (r1 == r2)
                {
                    return new Tuple<float, float>((float)50, (float)50); // tie
                }
                else
                {
                    return new Tuple<float, float>(r1-r2, /*r2-r1*/0); 
                }
            }

            return new Tuple<float, float>(0,0);
        }
    }

    public struct State
    {
        public bool _me;
        public bool _opponent;
        public bool _free { get { return !_me && !_opponent; } }
        public byte _stateId;
    }

    public struct Action
    {
        public double _value;
        public Byte _position; // [0-15]
        public Byte _rotation; // [0-7], where 1'st square(1 - left, 2 - right), 2'nd square(3 - left, 4 - right), 3'nd square(5 - left, 6 - right), 4'th square(7 - left, 8 - right)
        public Byte _explorations;

        public int ConvertToInt()
        {
            return _position * 8 + _rotation - 1;
        }
    }

    public class StateSpace
    {
        QTable _qfunction = new QTable();
        //QFunction _qfunction = new QFunction();
        //ANN _qfunction = new ANN();

        Reward _reward = new Reward();
        List<int> positions = new List<int>();
        static byte i = 0;
        State[] stateSpace = Enumerable.Repeat(1, 16).Select(x => new State {_stateId = i++ }).ToArray();

        QSpace qSpace = new QSpace();

        List<Winner> _winningPos = new List<Winner> 
        {
            new Winner{ _pos = new List<int>{0,5,10,15}, _id = 1},
            new Winner{ _pos = new List<int>{3,6,9,12}, _id = 2},
            new Winner{ _pos = new List<int>{0,1,2,3}, _id = 3},
            new Winner{ _pos = new List<int>{4,5,6,7}, _id = 4},
            new Winner{ _pos = new List<int>{8,9,10,11}, _id = 5},
            new Winner{ _pos = new List<int>{12,13,14,15}, _id = 6},
            new Winner{ _pos = new List<int>{0,4,8,12}, _id = 7},
            new Winner{ _pos = new List<int>{1,5,9,13}, _id = 8},
            new Winner{ _pos = new List<int>{2,6,10,14}, _id = 9},
            new Winner{ _pos = new List<int>{3,7,11,15}, _id = 10},
        };

        public void Write(TextWriter writer)
        {
            _qfunction.Write(writer);
        }

        public void Load(TextReader reader)
        {
            _qfunction.Load(reader);
        }

        public void LearnQNetwork()
        {
          //  _qfunction.LearnQNetwork();
        }

        public void ResetState()
        {
            i = 0;
            qSpace.ResetState();
        }


        public void SelfTrain2(List<System.Windows.Forms.Button> buttons, int episodes)
        {
            int ngames = 75;
            int player = 0;
            int[] wins = new int[2];
            bool showPlay = false;
            double T1 = 10, T2 = 10;
            List<double> winningRation = new List<double>();
            var rand = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode())).Value;
            Dictionary<Transition,bool> qActionsHistory = new Dictionary<Transition,bool>();

            //_qfunction.Test();
            
            for (int y = 1; y < episodes; y++)
            {

                double err = 0.0;

                //temperature -= 10;
                int games = 0;

                do
                {
                    qSpace.ResetState();
                    player = 0;
                    int playerId;
                    List<Transition> gameHistory = new List<Transition>();

                    do
                    {
                        if (qSpace.IsFullBoard()) break;

                        playerId = player % 2;

                        // Q(S,A), S->A-> observe(S')
                        QSpace s1, s3;

                        s1 = qSpace.Clone();
                        var action = _qfunction.BoltzmannSelection(qSpace, (player % 2 == 0)? T1 : T2); // second player is always more random
                        qSpace.TakeAction(action);

                        

                            /*if (gameHistory.Count > 1)
                            {
                                gameHistory[gameHistory.Count - 2].nextState = s1.Clone();
                                gameHistory[gameHistory.Count - 2].nextAction = action;
                            }*/

                            //var reward = _reward.GetReward(qSpace);
                            var reward = _qfunction._neuralNet.GetReward(qSpace);
                            //games++;

                        // check if it is terminal state
                        if (reward.Item1 != 0)
                        {
                            //if (player % 2 == 0)
                            /*{
                                ViewState(buttons);
                            }*/
                            //_qfunction._neuralNet.MakeInputVector(qSpace);

                            var transition = new Transition { state = s1, action = action, nextState = null, reward = reward.Item1 };
                            gameHistory.Add(transition);
                            
                            // observe state
                            //gameHistory[gameHistory.Count - 2].nextState = null;
                            gameHistory[gameHistory.Count - 2].reward = reward.Item2;

                            if (reward.Item1 != reward.Item2 && reward.Item1 != 0)
                            {
                                wins[(reward.Item1 > reward.Item2) ? player % 2 : (player + 1) % 2]++;
                            }
                            else
                            {
                                // tie
                                wins[0]++;
                                wins[1]++;

                            }
                            games++;

                            err += _qfunction.UpdateEpisodeQLearning(gameHistory);
                            gameHistory.Clear();
                            break;
                        }

                        // store S,a
                        gameHistory.Add(new Transition { state = s1, action = action, nextState = null, reward = (player % 2 == 0) ? 0: 0 });

                        // observe state for previous player
                        //s3 = qSpace.Inverse(); // opponent will observe it
                        qSpace = qSpace.Inverse(); // opponent will observe it

                        //if (gameHistory.Count > 1)
                        //{
                        //    gameHistory[gameHistory.Count - 2].nextState = s3;
                        //}

                        //if (showPlay && player % 2 == 0)
                        //{
                        //    ViewState(buttons);
                        //}


                        //if (showPlay && player % 2 == 1)
                        //{
                        //    ViewState(buttons);
                        //}


                        player++;
                    }
                    while (true);
                }
                while(games < ngames);

                winningRation.Add(err / games);

                Debug.WriteLine(string.Format("Ratio:{0}, Error: {1}", (double)wins[1] / wins[0], Math.Sqrt(err / games)));

                wins[0] = 0;
                wins[1] = 0;

                _qfunction.UpdateQNetwork();

                if (y % 100 == 0)
                {
                    Debug.WriteLine("Saving network....");
                    using (System.IO.FileStream fs = File.OpenWrite(@"D:\TFS\TP\SandBox\MachineLearning\OLS\PentagoSharp\bin\Debug\StateWeightsFUILL_Filter2.txt"))
                    {
                        using (var writer = new StreamWriter(fs))
                        {
                            Write(writer);
                        }
                    }
                }

            }
            qSpace.ResetState();
        }

        /*double UpdateGame(List<Transition> qActionsHistory)
        {
            double e = 0.0;
            qActionsHistory.Reverse();

            qActionsHistory.ForEach(transition =>
            {
                double nextStateValue = (transition.nextState != null) ?  _qfunction.GetMaxAction(transition.nextState).Value : 0;
                double qValue = transition.reward + _qfunction._gamma * nextStateValue;
                double error = qValue - transition.action.Value;
                _qfunction.QUpdate(error, transition.action, transition.state);
                e += error * error;
            });
            return e;
        }*/

        public void ViewState(List<System.Windows.Forms.Button> buttons)
        {
            Action<QSpace, int, System.Windows.Forms.Button> viewState = (QSpace s, int i, System.Windows.Forms.Button b) =>
            {
                if (s.IsMe(i))
                {
                    b.Text = "X";
                    b.Enabled = false;
                }
                else if (s.IsOpponent(i))
                {
                    b.Text = "0";
                    b.Enabled = false;
                }
                else
                {
                    b.Text = "";
                    b.Enabled = true;
                }

                b.Refresh();
            };

            //SwitchToOponent();
            for(int i = 0; i < 36; ++i)
            {
                viewState(qSpace,i,buttons[i]);
            }
        
        }

        public bool AcquireMove(ActionMove action, List<System.Windows.Forms.Button> buttons)
        {
            qSpace.TakeAction(action);
            ViewState(buttons);
            var reward = _reward.GetReward(qSpace);
            qSpace = qSpace.Inverse();
            return false;
            return reward.Item1 != 0;
        }

        public bool PredictMove(List<System.Windows.Forms.Button> buttons, ref ActionMove action, int playerId)
        {
            //qSpace = qSpace.Inverse();

            //SwitchToOponent();

            action = _qfunction.BoltzmannSelection(qSpace, 1);

            qSpace.TakeAction(action);

            var reward = _reward.GetReward(qSpace);
            if (reward.Item1 > 0)
            {
                qSpace = qSpace.Inverse();
                ViewState(buttons);
                return true;
            }

            qSpace = qSpace.Inverse();
            ViewState(buttons);
            return false;
        }

        void RotateLeft(int square)
        {
            byte x1 = (new List<byte> { 0, 2, 8, 10 })[square];

            State tmp1 = stateSpace[x1];
            State tmp2 = stateSpace[x1 + 1];
            State tmp3 = stateSpace[x1 + 4];
            State tmp4 = stateSpace[x1 + 5];

            stateSpace[x1] = tmp2;
            stateSpace[x1 + 4] = tmp1;
            stateSpace[x1 + 5] = tmp3;
            stateSpace[x1 + 1] = tmp4;

            stateSpace[x1]._stateId = x1;
            stateSpace[x1 + 4]._stateId = (byte)(x1 + 4);
            stateSpace[x1 + 5]._stateId = (byte)(x1 + 5);
            stateSpace[x1 + 1]._stateId = (byte)(x1 + 1);
        }

        void RotateRight(int square)
        {
            byte x1 = (new List<byte> { 0, 2, 8, 10 })[square];

            State tmp1 = stateSpace[x1];
            State tmp2 = stateSpace[x1 + 1];
            State tmp3 = stateSpace[x1 + 4];
            State tmp4 = stateSpace[x1 + 5];

            stateSpace[x1 + 1] = tmp1;
            stateSpace[x1 + 5] = tmp2;
            stateSpace[x1 + 4] = tmp4;
            stateSpace[x1] = tmp3;

            stateSpace[x1]._stateId = x1;
            stateSpace[x1 + 4]._stateId = (byte)(x1 + 4);
            stateSpace[x1 + 5]._stateId = (byte)(x1 + 5);
            stateSpace[x1 + 1]._stateId = (byte)(x1 + 1);
        }
    }

    public class ANN
    {
        //List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(3, 2), new Tuple<int, int>(2, 3) };
        //public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(128, 48), new Tuple<int, int>(128, 128), new Tuple<int, int>(128, 128) };
        //public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(128, 48) };
        //public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(48, 48), new Tuple<int, int>(128, 48), new Tuple<int, int>(128, 128) };

        //public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(128, 48), new Tuple<int, int>(128, 128), new Tuple<int, int>(128, 128) };
        
        public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(32, 48), new Tuple<int, int>(32, 32), new Tuple<int, int>(128, 32) };

        //public List<Tuple<int, int>> hidden = new List<Tuple<int, int>> { new Tuple<int, int>(256, 48), new Tuple<int, int>(128, 256) };
        

        double learningRate = 0.05;
        double regularizationL2 = 0.01;
        public const  int players = 3;
        public bool testDerrivative = false;
        //public List<List<Vector<double>>> playerPredictionLayer = new List<List<Vector<double>>>(Enumerable.Repeat<List<Vector<double>>>(null, 300));// {  null, null, null, null, null, null, };
        //public Action[] playerActions = (Enumerable.Repeat<Action>(new Action(), 300)).ToArray();

        public Matrix<double>[] wHidden;
        public Vector<double>[] wHiddenBias;
        public Vector<double> lastLayerDerr;

        public ANN()
        {
            double r = 4.0 * Math.Sqrt(6.0 / (hidden[0].Item2 + hidden[hidden.Count - 1].Item1));
            Random rand = new Random(1234);

            wHidden = hidden.Select(x => Matrix<double>.Build.Dense(x.Item1, x.Item2, Enumerable.Repeat<double>(0.0, x.Item1*x.Item2).Select(z=>(rand.NextDouble() * 2 - 1) * r).ToArray())).ToArray();
            wHiddenBias = hidden.Select(x => Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, x.Item1).Select(z => (rand.NextDouble() * 2 - 1) * r).ToArray())).ToArray();

            wHidden[hidden.Count - 1] = Matrix<double>.Build.Dense(hidden.Last().Item1, hidden.Last().Item2,                
                Enumerable.Repeat<double>(0.0, hidden.Last().Item1*hidden.Last().Item2).Select(z => rand.NextDouble() / hidden.Last().Item1).ToArray());

            wHiddenBias[hidden.Count - 1] = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, hidden.Last().Item1).Select(z => rand.NextDouble() / hidden.Last().Item1).ToArray());

            lastLayerDerr = Vector<double>.Build.Dense(Enumerable.Repeat<double>(1.0, hidden[hidden.Count - 1].Item1).ToArray());
        }

        Vector<double> sigmoid(Vector<double> input, int layer)
        {
            if (layer == wHidden.Length - 1)
            {
                return input;
            }
            else
            {
                //return Vector<double>.Build.DenseOfArray(input.Select(x=> 1.0 /(1.0+Math.Exp(-x))).ToArray()); // sigmoid
                return Vector<double>.Build.DenseOfArray(input.Select(x => Math.Max(0, x)).ToArray()); // reLu
            }
        }

        Vector<double> derrivative(Vector<double> x, int layer)
        {
            if (layer == wHidden.Length)
            {
                return lastLayerDerr;
            }
            else
            {
                //return x.PointwiseMultiply((1.0 - x)); //sigmoid
                return Vector<double>.Build.DenseOfArray(x.Select(z => (z > 0) ? 1 : 0.001 * z).ToArray()); // reLu
            }
        }

        public Action GetMaxAction(State[] states)
        {
            Action maxAction = new Action { _value = 0.0 };
            var input = MakeInputVector(states);
            var output = Predict(input).Last();

            foreach (var state in states.Where(x => x._free))
            {
                for (int r = 0; r < 8; ++r)
                {
                    var qvalue = output[state._stateId * 8 + r];
                    if (maxAction._value < qvalue)
                    {
                        maxAction._value = qvalue;
                        maxAction = new Action { _position = state._stateId, _rotation = (byte)(r + 1), _value = qvalue };
                    }
                }
            }

            return maxAction;
        }

        public Tuple<Matrix<double>[], Vector<double>[]> QUpdate(double newVal, int actionId, State[] states, ref double? error)
        {
            var input = MakeInputVector(states);
            var outputs = MakeOutputs(input);
            error = newVal - outputs.Last()[actionId];
            return LearnDelta((double)-error, actionId, outputs);
        }

        public Action GetRandomAction(Random rand, State[] currentStates)
        {
            List<Action> ret = new List<Action>(16 * 8);
            Action maxAction = new Action { _value = 0.0 };

            var input = MakeInputVector(currentStates);
            var output = Predict(input).Last();

            var states = currentStates.Where(x => x._free).OrderBy(x => rand.NextDouble());

            if (states.Count() > 0)
            {
                State state = states.First();
                int rotation = (int)(rand.NextDouble() * 8);
                if (rotation == 8)
                {
                    throw new Exception("random error");
                }
                var randomAction =  new Action { _position = state._stateId, _rotation = (byte)(rotation + 1), _value = output[state._stateId * 8 + rotation] };
                return randomAction;
            }
            else
            {
                return new Action();
            }
        }

        public double GetQvalue(int actionId, State[] states)
        {
            var input = MakeInputVector(states);
            var output = Predict(input).Last();
            return output[actionId];
        }

        List<Vector<double>> GetQvalues(int actionId, State[] states)
        {
            var input = MakeInputVector(states);
            var output = Predict(input);
            return output;
        }

        /*public void QUpdate(double reward, int playerId)
        {
            double error = reward - playerActions[playerId]._value;
            int actionId = playerActions[playerId].ConvertToInt();
            Learn(-error, actionId, playerId);
        }*/

        

        Vector<double> MakeInputVector(State[] states)
        {
            Vector<double> input = Vector<double>.Build.Dense(48);

            for (int i = 0; i < 16 * 3; ++i)
            {
                State state = states[i / 3];
                if (i % 3 == 0)
                {
                    input[i] = (state._me) ? 1.0 : 0;
                }
                else if (i % 3 == 1)
                {
                    input[i] = (state._opponent) ? 1.0 : 0;
                }
                else
                {
                    input[i] = (state._free) ? 1.0 : 0;
                }

            }
            return input;
        }


        

        public List<Vector<double>> MakeOutputs(Vector<double> input)
        {
            List<Vector<double>> out_net = new List<Vector<double>> { input };
            for (int h = 0; h < wHidden.Length; h++)
            {
                var tnet_h = wHidden[h] * out_net[h] + wHiddenBias[h];
                out_net.Add(sigmoid(tnet_h,h));
            }
            return out_net;
        }

        /*public Vector<double> MakeDerrivative(int layer, int row, int col, bool bias, List<Vector<double>> outputs)
        {
            int bpLayer = outputs.Count - 2;
            Matrix<double> w = null;
            for (; bpLayer > layer; bpLayer--)
            {
                var dD = derrivative(outputs[bpLayer+1]);
                var ww = wHidden[bpLayer].PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(dD,hidden[bpLayer].Item2)));
                w = (w == null) ? ww : w * ww;
            }

            // this is the layer we take a derrivative
            Matrix<double> dW = Matrix<double>.Build.Dense(hidden[bpLayer].Item1,hidden[bpLayer].Item2, 0.0);
            dW[row, col] = 1.0;

            var d1 = derrivative(outputs[bpLayer + 1]);
            dW = dW.PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(d1, hidden[bpLayer].Item2)));
            var derr = ((w == null) ? dW : (w * dW)) * ((!bias) ? outputs[layer] : Vector<double>.Build.Dense(hidden[bpLayer].Item2,1.0));
            return derr;
        }*/

        public Matrix<double> MakeLayerDerrivatives(int layer, List<Vector<double>> outputs, int ioutput)
        {
            int bpLayer = outputs.Count - 2;
            Matrix<double> w = null;
            for (; bpLayer > layer; bpLayer--)
            {
                var dD = derrivative(outputs[bpLayer + 1], bpLayer + 1);
                var ww = wHidden[bpLayer].PointwiseMultiply(Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(dD, hidden[bpLayer].Item2)));
                w = (w == null) ? ww : w * ww;
            }

            // this is the layer we take a derrivative
            Matrix<double> ret = Matrix<double>.Build.Dense(hidden[bpLayer].Item1, hidden[bpLayer].Item2 + 1);
            var d1 = derrivative(outputs[bpLayer + 1], bpLayer + 1);
            var v1 = Vector<double>.Build.Dense(hidden[bpLayer].Item2, 1.0);
            var o1 = Vector<double>.Build.Dense(hidden[bpLayer].Item1, 0.0);
            var m1  = Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat<Vector<double>>(d1, hidden[bpLayer].Item2));

            //Matrix<double> dW = Matrix<double>.Build.Dense(hidden[bpLayer].Item1, hidden[bpLayer].Item2, 0.0);
            for (int row = 0; row < hidden[bpLayer].Item1; ++row)
            {
                for (int col = 0; col < hidden[bpLayer].Item2; ++col)
                {
                    //dW[row, col] = m1[row,col];
                    o1[row] = m1[row,col] * outputs[layer][col]; // dW*input
                    var derr = (w != null) ? (w.Row(ioutput) * o1) : o1[ioutput];

                    /*if (ioutput == row)
                    {
                        derr -= regularizationL2 * wHidden[bpLayer][row, col]; // L2 regularization
                    }*/

                    //var W = ((w == null) ? dW : (w * dW));
                    //var derr = W.Row(ioutput) * outputs[layer];
                    //var derr = W * outputs[layer];
                    //ret[row, col] = derr[ioutput];
                    ret[row, col] = derr;

                    // and bias
                    if (col == 0)
                    {
                        o1[row] = d1[row];

                        var dBias = (w != null) ? (w.Row(ioutput) * o1) : o1[ioutput];

                        /*if (ioutput == row)
                        {
                            dBias -= regularizationL2 * wHiddenBias[bpLayer][row]; // L2 regularization
                        }*/

                        ret[row, hidden[bpLayer].Item2] = dBias;

                        //var W = ((w == null) ? dW : (w * dW));
                        //var dBias = W * v1;
                        //ret[row, hidden[bpLayer].Item2] = dBias[ioutput];
                    }

                    //dW[row, col] = 0;
                    o1[row] = 0.0;
                }
            }
            return ret;

/*            dW[row, col] = 1.0;
dW = dW.PointwiseMultiply(m1);
            var derr = ((w == null) ? dW : (w * dW)) * ((!bias) ? outputs[layer] : v1);
            return derr;*/
        }

        void MakeElementWiseMultiplication(Matrix<double> matrix, Vector<double> vector)
        {
            for (int row = 0; row < matrix.RowCount; row++)
            {
                var o = vector[row];

                for (int col = 0; col < matrix.ColumnCount; col++)
                {
                    matrix[row, col] *= o;
                }
            }
        
        }

        public Vector<double> MakeNumericDerrivative(int layer, int row, int col, bool bias, List<Vector<double>> outputs)
        {
            // numeric derrivative
            if (!bias)
            {
                double origin = wHidden[layer][row, col];
                double alpha = 0.00000001;
                wHidden[layer][row, col] = origin + alpha;
                var d1 = MakeOutputs(outputs[0]);
                wHidden[layer][row, col] = origin - alpha;
                var d2 = MakeOutputs(outputs[0]);
                var derr = (d1.Last() - d2.Last()) / (2 * alpha);

                // restore weight
                wHidden[layer][row, col] = origin;
                return derr;
            }
            else
            {
                double origin = wHiddenBias[layer][row];
                double alpha = 0.00000001;
                wHiddenBias[layer][row] = origin + alpha;
                var d1 = MakeOutputs(outputs[0]);
                wHiddenBias[layer][row] = origin - alpha;
                var d2 = MakeOutputs(outputs[0]);
                var derr = (d1.Last() - d2.Last()) / (2 * alpha);

                // restore weight
                wHiddenBias[layer][row] = origin;
                return derr;
            
            }
        }

        public void TestDerrivatives(Vector<double> qinput2)
        {
            return;

        /*    double r = 1.0;
            Random rand = new Random(123);
            var inputs = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, hidden[0].Item2).Select(z => ( (rand.NextDouble() > 0.5) ? 1.0 : 0)).ToArray());
            var targets = Vector<double>.Build.Dense(Enumerable.Repeat<double>(0.0, hidden[hidden.Count-1].Item1).Select(z => rand.NextDouble()*10).ToArray());

            int i = 0;
            while (true)
            {
                Vector<double> p = Predict(inputs,0);
                var err = p - targets;

                int ioputput = (int)((double)targets.Count * rand.NextDouble());

                Learn(err[ioputput], ioputput, 0);
                if (i % 100 == 0)
                {
                    Debug.WriteLine(string.Format("{0}:{1}", i, err.Select(x=> x*x).Sum()));
                }
                i++;
            }*/

        }

        List<Vector<double>> Predict(Vector<double> input)
        {
            return MakeOutputs(input);
        }

        public void Learn(double error, int ioutput, List<Vector<double>> outputs)
        {
            var weights = LearnDelta(error, ioutput, outputs);
            for(int i = 0; i < hidden.Count; i++)
            {
                wHidden[i] -= weights.Item1[i];
                wHiddenBias[i] -= weights.Item2[i];
            }
        }

        Tuple<Matrix<double>[],Vector<double>[]> LearnDelta(double error, int ioutput, List<Vector<double>> outputs)
        {
            
            int layer = 0;
            var wHiddenDelta = hidden.Select(x => Matrix<double>.Build.Dense(x.Item1, x.Item2,0.0)).ToArray();
            var wHiddenDeltaBias = hidden.Select(x => Vector<double>.Build.Dense(x.Item1, 0)).ToArray();
  

            Func<double, double> round = (double d) =>
                {
                    int dd = 1000000;
                    return (double)((int)d * dd) / dd;
                };

            var dWeights = hidden.Select(x =>
                {
                    var weights = Matrix<double>.Build.Dense(x.Item1, x.Item2, learningRate);
                    var bias = Vector<double>.Build.Dense(x.Item1, learningRate);

                    Matrix<double> dWeigths = MakeLayerDerrivatives(layer, outputs, ioutput);

                    for (int row = 0; row < x.Item1; row++)
                    {
                        //var derrB = MakeDerrivative(layer, row, 0, true, predictionLayer);
                        bias[row] *= dWeigths[row, x.Item2];

                        if (testDerrivative)
                        {
                            var derrTestB = MakeNumericDerrivative(layer, row, 0, true, outputs);
                            if (round(dWeigths[row, x.Item2]) != round(Math.Round(derrTestB[ioutput],5)))
                            {
                                throw new Exception("sss");
                            }
                        }

                        for (int col = 0; col < x.Item2; col++)
                        {
                            //var derrW = MakeDerrivative(layer, row, col, false, predictionLayer);
                            weights[row, col] *= dWeigths[row, col];

                            // test derrivative
                            if (testDerrivative)
                            {
                                var derrTestW = MakeNumericDerrivative(layer, row, col, false, outputs);
                                if (round(dWeigths[row, col]) != round(derrTestW[ioutput]))
                                {
                                    throw new Exception("sss");
                                }
                            }
                        }
                    }

                    layer++;
                    return new Tuple<Matrix<double>, Vector<double>>(weights, bias);
                }).ToArray();

            for(int i = 0; i < hidden.Count; i++)
            {
                wHiddenDelta[i] += error * dWeights[i].Item1;
                wHiddenDeltaBias[i] += error * dWeights[i].Item2;
            }

            return new Tuple<Matrix<double>[],Vector<double>[]>(wHiddenDelta, wHiddenDeltaBias);
        }

    }

}


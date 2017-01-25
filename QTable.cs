using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Pentago
{
    public class Transition
    {
        public QSpace state;
        public ActionMove action;
        public QSpace nextState;
        public ActionMove nextAction;
        public double reward;
    }

    public class QTable
    {
        public NeuralNet _neuralNet = new NeuralNet();
        Dictionary<QSpace, Dictionary<int,ActionMove>> _qValue = new Dictionary<QSpace, Dictionary<int, ActionMove>>();
        Dictionary<QSpace, Dictionary<int, double>> _qValueError = new Dictionary<QSpace, Dictionary<int, double>>();

        Dictionary<QSpace, int> _qValueUpdated = new Dictionary<QSpace, int>();

        public double _alpha = 0.3;
        public double _gamma = 0.9;
        public int _exploredStates = 0;

        public QTable()
        {
        }

        #region serialize

        public void Write(TextWriter writer)
        {
            writer.WriteLine(_alpha);
            writer.WriteLine(_alpha);
            writer.WriteLine(_gamma);
            writer.WriteLine(_exploredStates);

            _neuralNet.Write(writer);

            /*writer.WriteLine(_qValue.Count);

            foreach (var qvalue in _qValue)
            {
                // save QSpace
                qvalue.Key.Write(writer);
                writer.WriteLine(qvalue.Value);
            }*/
        }

        public void Load(TextReader reader)
        {
            _alpha = double.Parse(reader.ReadLine());
            _alpha = double.Parse(reader.ReadLine());
            _gamma = double.Parse(reader.ReadLine());
            _exploredStates = int.Parse(reader.ReadLine());

            _neuralNet.Load(reader);

            /*int count = int.Parse(reader.ReadLine());

            for (int i = 0; i < count; ++i)
            {
                var qSpace = QSpace.Load(reader);
                //_qValue[qSpace] = double.Parse(reader.ReadLine());
            }*/
        }

        #endregion

        #region Q Updating

        /// <summary>
        /// Update q table value
        /// </summary>
        /// <param name="error"></param>
        /// <param name="action"></param>
        /// <param name="stateSpace"></param>
        public void QUpdate(double error, ActionMove action, QSpace stateSpace)
        {
            var actions = _qValue[stateSpace];
            actions[action.Id].Value += _alpha * error;

            //            var state = stateSpace.Clone();
            //            state.TakeAction(action);
            //            _qValue[state] +=  _alpha * error;

            if (!_qValueError.ContainsKey(stateSpace))
            {
                _qValueError[stateSpace.Clone()] = new Dictionary<int, double>();
                _qValueError[stateSpace][action.Id] = error;
            }

            //_qValueError[stateSpace][action.Id] = error;

            if (!_qValueUpdated.ContainsKey(stateSpace))
            {
                _qValueUpdated[stateSpace.Clone()] = 1;
            }
            else
            {
                _qValueUpdated[stateSpace]++;
            }

            //_qValueDiscovered[stateSpace] = error;
        }


        // <summary>
        ///  Perform Q-learning updating strategy
        /// </summary>
        /// <param name="episode"></param>
        /// <returns></returns>
        public double UpdateEpisodeQLearning(List<Transition> episode)
        {
            double error;
            double e = 0.0;

            var white = episode[episode.Count - 1];
            error = white.reward - white.action.Value;
            QUpdate(error, white.action, white.state);
            e += error * error;


            for (int i = episode.Count - 2; i >= 0; i--)
            {
                var reward = episode[i].reward;
                var state = episode[i].state;
                var action = episode[i].action;

                var nextState = state.Clone();
                nextState.TakeAction(action);
                nextState = nextState.Inverse();

                var qValue = reward - _gamma * GetMaxAction(nextState).Value;
                error = qValue - action.Value;
                QUpdate(error, action, state);
                e += error * error;
            }

            return e;
        }

        double learnThreshold = 0.5;

        /// <summary>
        /// Update QNetwork with sampled Q table values
        /// </summary>
        public void UpdateQNetwork()
        {
            double maxIterations = 20;
            double converge = 5.0; 
            int iterations = 0;
            double err = 0.0;

            var topErrorStates = _qValueError.ToDictionary(x => x.Key, x => x.Value
                    .Where(v => Math.Abs(v.Value) > Math.Abs(learnThreshold)).ToDictionary(a => a.Key, a => a.Value))
                    .Where(x => x.Value.Count > 0)
                    .ToDictionary(a => a.Key, a => a.Value);


            var trainingSample = _qValue.Where(x => topErrorStates.ContainsKey(x.Key)).ToDictionary(x => x.Key, x => x.Value
                                .Where(v => topErrorStates[x.Key].ContainsKey(v.Key))
                                .ToDictionary(v => v.Key, v => v.Value));

            Dictionary<QSpace, double> vs = new Dictionary<QSpace, double>();

            foreach (var ss in trainingSample)
            {
                foreach (var a in ss.Value.Values)
                {
                    var s = ss.Key.Clone();
                    s.TakeAction(a);

                    if (vs.ContainsKey(s))
                    {
                        if (Math.Abs(a.Value) > Math.Abs(vs[s]))
                        {
                            vs[s] = a.Value;
                        }
                    }
                    else
                    {
                        vs[s] = a.Value;
                    }
                }
            }

            foreach (var ss in trainingSample)
            {
                foreach (var a in ss.Value.Values)
                {
                    var s = ss.Key.Clone();
                    s.TakeAction(a);
                    a.Value = vs[s];
                }
            }


            if (topErrorStates.Count != _qValueError.Count)
            {
                Debug.WriteLine(string.Format("QNet Updating {0} of {1}", topErrorStates.Count,  _qValueError.Count ));
            }

                do
                {
                    err += _neuralNet.BatchUpdate(trainingSample, 32);
                    iterations++;

                    if (iterations % maxIterations == 0)
                    {
                        err /= maxIterations;
                        Debug.WriteLine(string.Format("QNet error: {0}", err));
                        if (err <= converge) break;
                        err = 0.0;
                    }
                }
                while (iterations < maxIterations);

            // forget cached Q table values. from now one these q values will be initialized by QNetwork
            _qValue.Clear();
            _qValueError.Clear();
            _neuralNet._inputCache.Clear();
        }

        #endregion

        #region Action(s) selection

        /// <summary>
        /// Draw an action from action distribution per state. QNetwork is used to initialize an action value.
        /// </summary>
        /// <param name="stateSpace"></param>
        /// <param name="temperature"></param>
        /// <returns></returns>
        public ActionMove BoltzmannSelection(QSpace stateSpace, double temperature)
        {
            var rand = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode())).Value;

            // get actions distribution
            var actions = GetActionValues(stateSpace,rand);
            var maxActionVal = actions.Max(x=>x.Value);
            var sumActions = actions.Sum(x=> Math.Exp((x.Value - maxActionVal)/ temperature));

            // draw an action
            var iid = rand.NextDouble();
            double probSum = 0.0;

            foreach (var a in actions)
            {
                probSum += Math.Exp((a.Value - maxActionVal) / temperature) / sumActions;
                if (probSum >= iid)
                {
                    /*if (!_qValue.ContainsKey(stateSpace))
                    {
                        _exploredStates++;
                        var stateActions = new Dictionary<int, ActionMove>();
                        stateActions.Add(a.Id,a);
                        _qValue[stateSpace.Clone()] = stateActions;
                    }*/
                    return a;
                }
            }

            throw new Exception("No Selection");
        }

        /// <summary>
        /// Retrieve maximum action move position
        /// </summary>
        /// <param name="stateSpace"></param>
        /// <returns></returns>
        public ActionMove GetMaxAction(QSpace stateSpace)
        {
            return _qValue[stateSpace].Values.OrderByDescending(x => x.Value).First();


            /*double val = double.MinValue, tmp = 0.0 ;
            int pos = -1, rotate = -1;
            var actionValues = _neuralNet.Predict(stateSpace);

            for (int i = 0; i < 16; i++)
            {
                //var stateMove = stateSpace.Clone();
                if (!stateSpace.IsFree(i))
                {
                    continue;
                }

                //stateMove.TakeMoveAction(i);
                for (int r = 1; r <= 8; r++)
                {
                    tmp = actionValues[i * 8 + r - 1];

                    if (val <= tmp)
                    {
                        val = tmp;
                        pos = i;
                        rotate = r;
                    }
                }
            }*/

            /*_transitions[stateSpace].ForEach(x =>
            {
                double qVal;

                if (_qValue.ContainsKey(x.Item1))
                {
                    qVal = _qValue[x.Item1];
                }
                else
                {
                    qVal = _neuralNet.Predict(x.Item1);
                    _qValue[x.Item1] = qVal;
                }

                if (val <= qVal)
                {
                    val = tmp;
                    pos = x.Item2.PositionsId;
                    rotate = x.Item2.Rotation;
                }

            });*/

            /* for (int i = 0; i < 36; i++)
             {
                 var stateMove = stateSpace.Clone();
                 if (!stateMove.IsFree(i))
                 {
                     continue;
                 }

                 stateMove.TakeMoveAction(i);

                 for (int r = 1; r <= 8; r++)
                 {
                     var stateRotate = stateMove.Clone();
                     stateRotate.TakeRotateAction(r);

                     if (_qValue.ContainsKey(stateRotate))
                     {
                         tmp = _qValue[stateRotate];
                     }
                     else
                     {
                         tmp = _neuralNet.Predict(stateRotate); // _randLimit * rand.NextDouble();
                         //tmp = _rand.NextDouble();
                     }

                     if (val <= tmp)
                     {
                         val = tmp;
                         pos = i;
                         rotate = r;
                     }
                 }
             }
             */

            /*if (rotate == -1)
            {
                throw new Exception("Board error");
            }

            return new ActionMove { PositionsId = pos, Rotation = rotate,  Value = val };*/
        }

        List<ActionMove> GetActionValues(QSpace stateSpace, Random rand)
        {
            if (!_qValue.ContainsKey(stateSpace))
            {
                List<ActionMove> actions = new List<ActionMove>();
                var actionValues = _neuralNet.Predict(stateSpace);

                for (int i = 0; i < 36; i++)
                {
                    //var stateMove = stateSpace.Clone();
                    if (!stateSpace.IsFree(i))
                    {
                        continue;
                    }

                    //stateMove.TakeMoveAction(i);
                    for (int r = 1; r <= 8; r++)
                    {
                        actions.Add(new ActionMove { PositionsId = i, Rotation = r, Value = actionValues[i * 8 + r - 1] });
                    }
                }
                _qValue[stateSpace.Clone()] = actions.ToDictionary(x=>x.Id, x=> x);
            }

            return _qValue[stateSpace].Values.ToList();

            /*
            List<ActionMove> actions = new List<ActionMove>();
            List<Tuple<QSpace, ActionMove>> transitions = null;

            if (_transitions.ContainsKey(stateSpace))
            {
                transitions = _transitions[stateSpace];
            }
            else
            {
                transitions = new List<Tuple<QSpace, ActionMove>>();

                for (int i = 0; i < 16; i++)
                {
                    var stateMove = stateSpace.Clone();
                    if (!stateMove.IsFree(i))
                    {
                        continue;
                    }

                    stateMove.TakeMoveAction(i);

                    for (int r = 1; r <= 8; r++)
                    {
                        var stateRotate = stateMove.Clone();
                        stateRotate.TakeRotateAction(r);

                        transitions.Add(new Tuple<QSpace, ActionMove>(stateRotate, new ActionMove { PositionsId = i, Rotation = r }));
                    }
                }

                _transitions[stateSpace.Clone()] = transitions;
            }

            transitions.ForEach(x =>
            {
                double qVal;

                if (_qValue.ContainsKey(x.Item1))
                {
                    qVal = _qValue[x.Item1];
                }
                else
                {
                    qVal = _neuralNet.Predict(x.Item1);
                    _qValue[x.Item1] = qVal;
                 }
                actions.Add(new ActionMove { PositionsId = x.Item2.PositionsId, Rotation = x.Item2.Rotation, Value = qVal });

            });*/

            /*
            for (int i = 0; i < 36; i++)
            {
                var stateMove = stateSpace.Clone();
                if (!stateMove.IsFree(i))
                {
                    continue;
                }

                stateMove.TakeMoveAction(i);

                for (int r = 1; r <= 8; r++)
                {
                    double qVal = 0.0;
                    var stateRotate = stateMove.Clone();
                    stateRotate.TakeRotateAction(r);

                    if (_qValue.ContainsKey(stateRotate))
                    {
                        qVal = _qValue[stateRotate];
                    }
                    else
                    {
                        qVal = _neuralNet.Predict(stateRotate);
                        _qValue[stateRotate] = qVal;
                        //var qvalue = _rand.NextDouble();
                    }

                    actions.Add(new ActionMove { PositionsId = i, Rotation = r, Value = qVal });
                }
            }
*/

            //return actions;
        }

        //Dictionary<QSpace, List<Tuple<QSpace, ActionMove>>> _transitions = new Dictionary<QSpace, List<Tuple<QSpace, ActionMove>>>();

        #endregion
    }
}

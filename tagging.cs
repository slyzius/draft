using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApplication1
{
    
    public class NaiveBayesian
    {
        FeatureManager _ftm;
        Dictionary<string, int> _categCounts;
        Dictionary<string, Dictionary<int,double>> _featureLikellihood;

        Dictionary<int, double> _userPrior;
        Dictionary<int, Dictionary<string, double>> _categUserLikellihood;
        int _totalVouchers;


        public void Initialize(Voucher[] inVouchers, FeatureManager ftm)
        {
            _ftm = ftm;

            var vouchers = ftm.ReadUniqueVouchers(inVouchers);

            _totalVouchers = vouchers.Length;

            // feature per category likellihood
            _categCounts = vouchers.GroupBy(x => x.TagName).ToDictionary(x => x.Key, x => x.Count());

            _featureLikellihood = vouchers.GroupBy(g => g.TagName)
                    .SelectMany(y =>y.SelectMany(c => _ftm.ReadFeatures(c.OcrFeatures.ToList()).Select(f=>new Tuple<string,int>(y.Key,f.Item1))))
                    .GroupBy(x=>x.Item1)
                    .ToDictionary(x=>x.Key,x=>x.GroupBy(y=>y.Item2).ToDictionary(z =>/*_ftm._featureById[*/z.Key/*]*/, z=> ((double)z.Count() + 1) / (x.Count() + _ftm._featureById.Count)));


            // user per category likellihood: p(categ,user) = p(categ|user)*p(user)
            int totalUsers = vouchers.Select(x => x.OrganizationId).Distinct().Count();
            _userPrior = vouchers.GroupBy(x => x.OrganizationId).ToDictionary(x => x.Key, x => ((double)x.Count() + 1) / (_totalVouchers + totalUsers));

            _categUserLikellihood = vouchers.GroupBy(g => g.OrganizationId)
                .ToDictionary(x => x.Key, x => x.GroupBy(y => y.TagName).ToDictionary(y => y.Key, y => ((double)y.Count() + 1) / (_categCounts[y.Key] + totalUsers)));

            //var tmp = vouchers.Where(x => x.OrganizationId == 3634).GroupBy(x => x.TagName).Select(x => new Tuple<string, double>(x.Key, ((double)x.Count() + 1) / (categCounts[x.Key] + totalUsers))).ToList();
        }

        public List<Tuple<string, double>> Predict(List<string> ocrFeatures, int organizationId)
        {
            var features = _ftm.ReadFeatures(ocrFeatures).ToDictionary(x=>x.Item1,x=>0);

            Dictionary<string, double> posterior = new Dictionary<string, double>();

            foreach (var category in _featureLikellihood)
            {
                if (_categUserLikellihood.ContainsKey(organizationId) && _categUserLikellihood[organizationId].ContainsKey(category.Key))
                {
                    posterior.Add(category.Key, Math.Log(_userPrior[organizationId] * _categUserLikellihood[organizationId][category.Key]));
                }
                else
                {
                    var unknownUserPrior = 1.0 / (_totalVouchers + _userPrior.Count + 1);
                    var unknownLikellihood = 1.0 / (_categCounts[category.Key] + _userPrior.Count + 1);
                    posterior.Add(category.Key, Math.Log(unknownUserPrior * unknownLikellihood));
                }

                //posterior.Add(category.Key, Math.Log(categPrior[category.Key]));

                foreach (var feature in _ftm._featureById)
                {
                    double likellihood;

                    if (category.Value.ContainsKey(feature.Key))
                    {
                        // we know the feature
                        likellihood = category.Value[feature.Key];
                    }
                    else
                    {
                        // feature is not used in category
                        // therfore apply not unknown feature likellihood
                        likellihood = 1.0 / (category.Value.Count + _ftm._featureById.Count + 1);
                    }

                    if (features.ContainsKey(feature.Key))
                    {
                        // existance likellihood
                        posterior[category.Key] += Math.Log(likellihood);
                    }
                    else
                    {
                        posterior[category.Key] += Math.Log(1.0 - likellihood);
                    }
                }
            }

            var tmp = posterior.Where(x => selectedFeatures[x.Key] > 0).Select(x => new Tuple<string, double>(x.Key, x.Value)).OrderByDescending(x => x.Item2).Take(3*2).ToList();
            return SelectPreferred(tmp);
            //return tmp;
        }

        private static List<Tuple<string, double>> SelectPreferred(IEnumerable<Tuple<string, double>> values)
        {
            double avg = values.Select(x => x.Item2).Sum() / values.Count();
            double sd = Math.Sqrt(values.Sum(x => Math.Pow(x.Item2 - avg, 2))/ values.Count());

            return sd > 0 ? values.Select(x => new Tuple<string, double>(x.Item1, (x.Item2 - avg) / sd)).Where(x => x.Item2 >= 0.4).ToList()
                : values.Select(x => new Tuple<string, double>(x.Item1, 1)).ToList();
        }

    }
}

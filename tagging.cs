 <!-- Generate NuGet package for .Net clients -->
    <Message Text="Generate NuGet package for .Net" />
    <Exec Command="&quot;$(SolutionRoot)\.nuget\nuget.exe&quot; pack &quot;$(SolutionRoot)\Source\Iit.Research.AceService\ACE.API.Proxy.Nuget\ACE.API.Proxy.Nuget.nuspec&quot; -Version $(Version) -BasePath &quot;$(SolutionRoot)\Bin\Client&quot; -OutputDirectory &quot;$(BinariesRoot)&quot; -Properties ReleaseNotes=&quot;&quot;"
          IgnoreExitCode="false" />


public class OrganizationPreferences
    {
        public Dictionary<Tuple<int, string>, double> _organizationTagLikelihood;
        public void Initialize(Voucher[] vouchers)
        {
            _organizationTagLikelihood = vouchers.GroupBy(x => x.OrganizationId)
               .SelectMany(x => x.GroupBy(y => y.TagName).Select(y => new Tuple<int, string, double>(x.Key, y.Key,((double)y.Count()) / x.Count())))
               .ToDictionary(x=> new Tuple<int, string>(x.Item1,x.Item2), x=> x.Item3);
        }
    }







public List<Tuple<string, double>> Predict2(Voucher voucher, FeatureManager ftm) 
        {
            int dim = ftm.NumberOfValidFeatures + 1;
            double[] inputs2 = new double[dim];

            var features = ftm.ReadWhenPredictingFeatures(voucher.OcrFeatures.ToList());
            features.ForEach(x =>
            {
                {
                    inputs2[x.Item1] = x.Item2;
                }
            });
 
            // clasify against each category
            var answers = _svm.Select(kvp => new Tuple<string, double>(kvp.Key, kvp.Value.Compute(inputs2)))
                .Where(x=>x.Item2 > -0.5)
                .ToArray();

                //.OrderByDescending(x => x.Item2).ToArray();

            if (answers.Length == 0)
            {
                return null;
            }

            var min = answers.Min(x => 1 + x.Item2);
            var max = answers.Max(x => 1 + x.Item2);

            if (min == max)
            {
                min = 0;
                max = 1;
            }

            var answers2 = answers.Select(x => 
                {
                    var likelihoodKey = new Tuple<int, string>(voucher.OrganizationId, x.Item1);
                    var likelihood = (_organizationPreferences._organizationTagLikelihood.ContainsKey(likelihoodKey)) ? _organizationPreferences._organizationTagLikelihood[likelihoodKey] : 0.0;
                    //var ret = new Tuple<string, double>(x.Item1, ((1 + x.Item2 - min) / (max - min)) * 0.6 + 0.4 * likelihood);
                    var ret = new Tuple<string, double>(x.Item1, x.Item2 * 0.8 + 0.2 * likelihood);
                    return ret;
                })
                .OrderByDescending(x => x.Item2)
                .ToArray();


            //return answers2.First().Item1;

            if (answers2.Length > 0)
            {
                var prediction = answers2.First();
                return prediction.Item1.Split(new char[] { ',' }).Select(x => new Tuple<string, double>(x, prediction.Item2)).ToList();
            }
            else
            {
                return null;
            }
        }

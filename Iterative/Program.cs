using System;
using System.Collections.Generic;

namespace Iterative_method_for_sequential_decision_processes
{
    class Program

    {
      
        static List<double> GaussianElimination(double[,] coefficients, int countEq)
        {
            int row = countEq;
            double d, s;

            double[,] A = new double[row, row];
            double[,] a = new double[row, row];

            double[] B = new double[row];
            double[] b = new double[row];
            double[] x = new double[row];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < row; j++)
                {
                    A[i, j] = coefficients[i, j];
                    a[i, j] = A[i, j];
                }
                B[i] = coefficients[i, countEq]; // Последний элемент в коэффициентах - свободный член
                b[i] = B[i];
            }

            for (int k = 0; k < row; k++)
            {
                for (int j = k + 1; j < row; j++)
                {
                    d = A[j, k] / A[k, k];
                    for (int i = k; i < row; i++)
                    {
                        A[j, i] = A[j, i] - d * A[k, i];
                    }
                    B[j] = B[j] - d * B[k];
                }
            }

            for (int k = row - 1; k >= 0; k--)
            {
                d = 0;
                for (int j = k; j < row; j++)
                {
                    s = A[k, j] * x[j];
                    d += s;
                }
                x[k] = (B[k] - d) / A[k, k];
            }

            List<double> Answer = new List<double>();
            for (int i = 0; i < row; i++)
            {
                Answer.Add(x[i]);
            }

            return Answer;
        }

        class State
        {
            private List<Strategy> Strategies = new List<Strategy>();

            public State()
            {
            }

            public List<Strategy> GetStrategies()
            {
                return Strategies;
            }

            public void AddStrategy(Strategy s)
            {
                Strategies.Add(s);
            }
        }

        class Strategy
        {
            private List<double> P = new List<double>();
            private double q;

            public Strategy(string[] str)
            {
                q = Convert.ToDouble(str[str.Length - 1]);
                for (int sn = 0; sn < str.Length - 1; sn++)
                    P.Add(Convert.ToDouble(str[sn]));
            }

            public double GetQ() 
            {
                return q;
            }

            public List<double> GetP()
            {
                return P;
            }

        }

        static void Main(string[] args)
        {
            // Считывание данных в формате:
            // Число состояний
            //   Число стратегий в состоянии
            //     Стратегия в формате p1 p2 ... pn q
            Console.WriteLine("Enter number of states:");
            int nstates = Convert.ToInt32(Console.ReadLine());
            List<State> States = new List<State>();
            for (int i = 0; i < nstates; i++)
            {
                Console.WriteLine("Enter number of strategies for " + (i + 1) + " state:");
                int nstrategies = Convert.ToInt32(Console.ReadLine());
                State bufstate = new State();
                for (int j = 0; j < nstrategies; j++)
                {
                    Console.WriteLine("Enter a " + (j + 1) + " strategy:");
                    string input = Console.ReadLine();
                    string[] words = input.Split(' ');
                    Strategy bufstrategy = new Strategy(words);
                    bufstate.AddStrategy(bufstrategy);
                }
                States.Add(bufstate);
            }

 
            List<int> d = new List<int>();
            List<Strategy> pq = new List<Strategy>();
            for (int i = 0; i < nstates; i++)
            {
                int nbest = 0;
                double qbest = States[i].GetStrategies()[0].GetQ();
                for (int j = 0; j < States[i].GetStrategies().Count; j++)
                {
                    if (States[i].GetStrategies()[j].GetQ() > qbest)
                    {
                        nbest = j;
                        qbest = States[i].GetStrategies()[j].GetQ();
                    }
                }
                d.Add(nbest);
                pq.Add(States[i].GetStrategies()[nbest]);
            }

            List<int> lastd = new List<int>();
            while (d.Equals(lastd))
            {
                // g + v1 = q1 + p11 * v1 + p12 * v2 |-----\ | g + (1 - p11) * v1 = q1
                // g + v2 = q2 + p21 * v1 + p22 * v2 |-----/ | g + (-p21) * v1 = q2
                double[,] coefficients = new double[2, 3];
                coefficients[0, 0] = 1;
                coefficients[0, 1] = 1 - pq[0].GetP()[0];
                coefficients[0, 2] = pq[0].GetP()[0];
                coefficients[1, 0] = 1;
                coefficients[1, 1] = -pq[0].GetP()[0];
                coefficients[1, 2] = pq[0].GetP()[0];
                List<double> gausAns = GaussianElimination(coefficients, coefficients.GetLength(0));
                double g = gausAns[0];
                double v1 = gausAns[1];
                double v2 = 0;

                lastd.Clear();
                for (int i = 0; i < d.Count; i++)
                    lastd.Add(d[i]);

                // Улучшение решения
                for (int i = 0; i < nstates; i++)
                {
                    double maxQ = Double.NegativeInfinity;
                    for (int j = 0; j < States[i].GetStrategies().Count; j++)
                    {
                        double param = States[i].GetStrategies()[j].GetQ() + States[i].GetStrategies()[j].GetP()[0] * v1;
                        if (param > maxQ)
                        {
                            d[i] = j;
                            maxQ = param;
                        }
                    }
                }

                for (int i = 0; i < States.Count; i++)
                    pq[i] = States[i].GetStrategies()[d[i]];
            }

            Console.WriteLine("Best solution:");
            for (int i = 0; i < d.Count; i++)
                Console.Write((d[i] + 1) + " ");
        }
    }
}

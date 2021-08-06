using System;
using System.Collections.Generic;

namespace Iterative
{
    class Program
    {
        class State
        {
            private List<Strategy> Strategies = new List<Strategy>(); // Список стратегий состояния

            public State() // Пустой конструктор
            {
            }

            public List<Strategy> GetStrategies() // Извлечение стратегий
            {
                return Strategies;
            }

            public void AddStrategy(Strategy s) // Добавление стратегии
            {
                Strategies.Add(s);
            }
        }

        class Strategy
        {
            private List<double> P = new List<double>(); // Вероятность перехода для данной стратегии
            private double q; // Ожидаемый базовый доход

            public Strategy(string[] str) // Конструктор стратегии
            {
                q = Convert.ToDouble(str[str.Length - 1]);
                for (int sn = 0; sn < str.Length - 1; sn++)
                    P.Add(Convert.ToDouble(str[sn]));
            }

            public double GetQ() // Извлечение ожидаемого базового дохода
            {
                return q;
            }

            public List<double> GetP() // Извлечение вероятностей перехода
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
                Console.WriteLine("Enter number of strategies for " + i + " state:");
                int nstrategies = Convert.ToInt32(Console.ReadLine());
                State bufstate = new State();
                for (int j = 0; j < nstrategies; j++)
                {
                    Console.WriteLine("Enter a " + j + " strategy:");
                    string input = Console.ReadLine();
                    string[] words = input.Split(' ');
                    Strategy bufstrategy = new Strategy(words);
                    bufstate.AddStrategy(bufstrategy);
                }
                States.Add(bufstate);
            }

            // Вычисление начального положения
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
    }
    }
}
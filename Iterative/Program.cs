using System;
using System.Collections.Generic;

namespace Iterative
{
    class Program
    {
        тестирование гита - +++++++++++++++++++++++++++++++++
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // СЧИТЫВАНИЕ ДАННЫХ
            Console.WriteLine("Enter number of states:");
            int nstates = Convert.ToInt32(Console.ReadLine());
            List<List<List<double>>> Strategies = new List<List<List<double>>>();
            List<List<double>> Profit = new List<List<double>>();
            for (int i = 0; i < nstates; i++)
            {
                List<List<double>> bufi = new List<List<double>>();
                List<double> bufq = new List<double>();
                Console.WriteLine("Enter number of strategies for " + i + " state:");
                int nstrategies = Convert.ToInt32(Console.ReadLine());
                for (int j = 0; j < nstrategies; j++)
                {
                    Console.WriteLine("Enter a " + j + " strategy:");
                    string input = Console.ReadLine();
                    string[] words = input.Split(' ');
                    List<double> bufj = new List<double>();
                    for (int k = 0; k < nstates; k++)
                        bufj.Add(Convert.ToDouble(words[k]));
                    bufq.Add(Convert.ToDouble(words[nstates]));
                    bufi.Add(bufj);
                }
                Profit.Add(bufq);
                Strategies.Add(bufi);
            }

            // первый шаг
            List<int> d = new List<int>();
            List<List<double>> P = new List<List<double>>();
            List<double> q = new List<double>();
            for (int i = 0; i < nstates; i++)
            {
                int nbest = 0;
                double qbest = Profit[i][0];
                for (int j = 0; j < Profit[i].Count; j++)
                {
                    if (Profit[i][j] > qbest)
                    {
                        nbest = j;
                        qbest = Profit[i][j];
                    }
                }
                d.Add(nbest);
                P.Add(Strategies[i][nbest]);
                q.Add(qbest);
            }
        }
    }
}
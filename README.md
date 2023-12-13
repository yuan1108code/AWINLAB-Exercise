Exercise1


Exercise2

超啟發式學習（Meta-heuristic Algorithm)

  (1)Hill Climbing:與模擬退火算法(Simulated Annealing)近似的爬山算法。隨機產生出一組合法解，然後去評估這組解的結果。藉由Transition的方式將進行更新，並Evalution以達到更好的效果。藉由多次的迭代，以達到逼近全域最佳解的結果。
   
  (2)Genetic Algorithm:
      基因遺傳演算法的進行以五個步驟為主: Initial population、Fitness function、Selection、Crossover、Mutation
      Initialization: 以隨機的方式產生多條染色體作為初始解。
      Fitness: 將初始的大量群集染色體解碼，帶入背包問題中，計算目標函數值。Fitness 越小代表具有較好的資質，將來被複製或選取為菁英個體的機率也較高。
      Selection: 從原來群組篩選出較佳的個體組成下一代族群，因此越高適應值的染色體，有較高的機率被選擇。
      Crossover: 當染色體需要進行交配的程序，便將隨機選取染色體，將其基因列重新的組合。
      Mutation: 當突變的機率低於事先定義的突變率，便會進行突變的程序。過程中搜尋的方式更為離散，以防止收斂在局部最佳值的情況。
    	
	輸入的資料: Weights、Profits、Capacity、Interation、pop_size(for GA)、num_parents_mating(for GA)
    輸出的資料: HC_profit_history & GA_profit_history 根據100次迭代的收斂情況
  


"""
    Criar uma nova collection com base na ground_truth_results
        - experiments (str[]): Deve armazenar os nomes de cada experimento com base em
            uma lista dos nomes dos experimentos que são enviados como parâmetro.
        - models (str[]): Deve armazenar o nome de cada modelo utilizado em cada experimento
            com base na experiments_ground_truth collection que deve ser filtrado 
            de acordo com a lista de experimentos enviada como parâmetro.
            Cada reigstro em experiments_ground_truth possui o campo experiment_gt_name
            que deve ser utilizado para filtrar os experimentos.

        - task_id (str)
        - instance_id (str) 
        - prompt_MTI (str) 
        - gt_prompt (str) 

    ### Campos de resposta
    // Actual state of the metric fields inside gt_MTI_answer and gt_STI_answer
    "coherence": {
      "score": X,
      "explanation": "a"
    },
    "specificity": {
      "score": X,
      "explanation": "b"
    },
    "informativeness": {
      "score": X,
      "explanation": "c"
    },
    "relevance": {
      "score": X,
      "explanation": "d"
    },
    "Understandability": {
      "score": X,
      "explanation": "e"
    }

    //Expect Result 
    "gt_MTI_answer": {
        "coherence": {
        "score_gpt-4o-mini-2024-07-18": X,
        "score_llama-3.3-70b-versatile": Y,
        "score_gpt-4o-2024-08-06": Z,
        "median_score": Deve ser a mediana dos 3 scores acima, calculada e armazenada aqui        
        "explanation_gpt-4o-mini-2024-07-18": "A"
        "explanation_llama-3.3-70b-versatile": "B"
        "explanation_gpt-4o-2024-08-06": "C"
        },
        ... (Assim sucessivamente para cada métrica dentro de gt_MTI_answer e gt_STI_answer)        
    },
    "gt_STI_answer": {
    ... Mesma lógica do gt_MTI_answer ...
    }

        

"""


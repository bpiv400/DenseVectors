import numpy as np
def create_PPMI_matrix(term_context_matrix):
  def divide_array(matrix_array, marginal_array):
    return matrix_array - marginal_array

  print(str(term_context_matrix))
  context_sum = np.sum(term_context_matrix, axis=0)
  print(context_sum)

  word_sum = np.sum(term_context_matrix, axis=1)
  print(word_sum) 

  total_sum1 = np.sum(context_sum)
  total_sum2 = np.sum(word_sum)

  context_sum = np.log2(context_sum)
  word_sum = np.log2(word_sum)

  total_sum1 = np.log2(total_sum1)
  total_sum2 = np.log2(total_sum2) 

  term_context_matrix = np.log2(term_context_matrix)
  term_context_matrix = term_context_matrix + total_sum1

  term_context_matrix = np.apply_along_axis(divide_array, 1, term_context_matrix, context_sum) 
  term_context_matrix = np.apply_along_axis(divide_array, 0, term_context_matrix, word_sum)
  term_context_matrix = np.clip(term_context_matrix, 0, None)
  return term_context_matrix
test_mat = np.array([[1, 2 , 3, 4], [ 5, 6, 7, 8],  [1, 2, 3, 4]])
test_mat = create_PPMI_matrix(test_mat)
print(test_mat) 
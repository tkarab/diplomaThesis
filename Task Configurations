**General formula for number of possible task configurations for a given experiment**

Suppose :
 - **s** total subjects
 - **g** total gestures
 - **r** total reps per gesture
 - **N** classes per task
 - **k** samples per task class
 - **1** query gesture per task, belongs to one of the N classes of that task

### Experiments 1, 2b, 3
The number of possible N-tuples of gestures chosen from the total g available is $\binom{g}{N} = \frac{g!}{N!(g-N)!}$
Given that one of the N chosen classes will also be chosen as the query one that gives us $\binom{N}{1}  = N$ possible query classes per chosen N-tuple therefore resulting in $N\times\binom{g}{N}$ possible N-tuples,query gesture that comprise the task

For a given set of N classes, chosen query class there $s\\_r = s\times r$ possible combinations of subject, rep for a given gesture number. There are N classes and for N-1 of those we need to chose k (s,r) pairs of the s_r available and from the query one we shall choose k+1.
Thie leads us to: 
 - $\binom{s\\_r }{k}$ possible combinations of samples for N-1 of the chosen classes
 - $\binom{s\\_r}{k+1}$ combinations for the chosen query class
 
Thus resulting in:
$\binom{s\\_r}{k}^{N-1}\times\binom{s\\_r}{k+1}$  total combinations over all classes for a given N-tuple of categories and a chosen query category.

For all possible N-tuples this will result in:
$N\times\binom{g}{N}\times\binom{s\\_r}{k}^{N-1}\times\binom{s\\_r}{k+1}$  possible tasks for these experiments

### Experiment 2a
Again the number of possible N-tuples of classes (+a chosen query class) is $N\times\binom{g}{N}$
This time for a given set of N classes, the samples are all chosen from the same subject. Therefore, for a given subject, for each class of the support set we need to choose k of the available r repetitions (g,s are constant). 
This is a total of $\binom{r}{k}$ possible combinatins of k samples per class for the non-query classes. For the query one the number will be $\binom{r}{k+1}$. This means that there are:
 - $\binom{r}{k}$ possible combinations of samples for the N-1 non-query classes
 - $\binom{r}{k+1}$ combinations of sample keys for the query class
 
In total:
$\binom{r}{k}^{N-1} \times\binom{r}{k+1}$ possible tasks for given subject number, N-tiple of classes and query class.
Given that any of the s subjects could be randomly chosen for a task this means that the total available tasks for a set of N classes are $s\times\binom{r}{k}^{N-1} \times\binom{r}{k+1}$

For all possible N-tuples this will result in:
$N \times \binom{g}{N} \times s \times\binom{r}{k}^{N-1} \times \binom{r}{k+1}$ possible tasks for experiment 2a

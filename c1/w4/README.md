# Machine Translation:

## WORD VECTORS + NEURAL NETWORKS:
We can use vector space for translation. Let us look at different methodologies to pull this off.
Transforming word vectors: 
`XR = Y`

So, let X be word vectors for english words and similarly let Y be the word vectors for French words.
Our aim would be reduce the distance between our predicted XR and Y, i.e. loss = Forbius(|| XR - Y ||)

We will be using Gradient Descent to update R.

Loss = `Mean Square Error`.

## ACTUAL TRANSLATION WOULD REQUIRE ABOVE RESULTS + KNN:
Since, we know that, there will always be an error between XR and Y, we will be using KNN to actually translate these specific words.

## HASH TABLES:
Let us replace each word with a particular number, however, we would need a function to assign this word that number.
As of now we have been replacing words by vectors, therefore,

`hash_function(vector) --> hash_value`

The most basic hash function would be:

```python
def basic_hash_table(value, n_buckets):
	def hash_function(value, n_buckets):
		return int(value)%n_buckets
	hash_table = {i:[] for i in range(n_buckets)}
	for value in value_l:
		hash_value = has_function(value, n_buckets)
		hash_table[hash_value].append(value)
	return hash_table
```

Ideally one would need a hash bucket which collects similar items in the same bucket, i.e. `locality sensitive hashing`.

### LOCALITY SENSITIVE HASHING:
We can use planes and their perpendicular vector to create locality sensitive hashing. Using Normal Vector:
The dot product between a given vector and the respective perpendicular vector tells us a lot about its locality,
1. +ve: Similar, to the normal
2. 0: same as the normal
3. -ve: Away from the normal

```python
def side_of_plane(P, v):
	dotproduct = np.dot(P, v.T)
	sign_of_dot_product = np.sign(dotproduct)
	sign_of_dot_product_scaler = np.asscalar(sign_of_dot_product)
```

#### What if multiple planes are present:
We use multiple planes and calculate the signs w.r.t each of the planes, now we have a vector of lenght n_of_hash_planes, 
we represent them in the following way,
* sign[i] >= 0, h[i] = 1
* sign[i] <  0, h[i] = 0

hash = summation(2^sign[i]), where, i from 0 to n_of_hash_planes.

```python
def hash_multiple_plane(P_l, v):
	hash_value = 0
	for i, P in enumerate(P_l):
		sign = side_of_plane(P, v)
		hash_i = 1 if sign >= 0 else 0
		hash_value += 2**i * hash_i
	return hash_value
```

### Approximating KNN:

```python
num_dimensions = 2
num_planes = 3

random_planes_matrix = np.random.normal(size=(num_planes, num_dimension))
```

Let `v = np.array([[1, 3]])`

Use the basic side_of_plane_matrix to find out the sides w.r.t different planes, and then generate hash values, w.r.t each of them. Since, local values will automatically, be closer to each other KNN will be more efficient and faster, but at the cost accuracy.

## DOCUMENT SEARCH (EMBEDDING):
A method of converting documents to embeddings, is taking the summation of its unique word vectors, thereby, allowing much greater understanding.
One vectorized KNN can be used to find the Nearest Neighbours.
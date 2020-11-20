# Vector Space:

"You shall know a word by the company it keeps". You can get context of documents using this method.
It captures the relative meaning of words.

## Word by Word design:
Number of times two words occur togeather within a distance `k`.
Thereby, one can create a co-occurrence matrix.

## Word by Document Design:
Number of times the word is found in the particular document.

#### Both of these lead to generation of a vector space.

## Looking at Similarities:
### 1. Euclidean Distance:
Can be used to calculate the similarity between two documents, there by allowing us cluster similar documents.
The distance can be calculated by taking the norm between the two n-dimensional spaces.

code: `d = np.linalg.norm(a-b)`

### 2. Cosine Similarity:
Problems with Euclidean Distance:
1. Number of samples can influence final distance results.
2. Using cosine allows us to see a more intuition friendly approach to finding similar documents

Finding the cosine angle between two vectors, allows us to find similarity between two vectors.

`cosA belongs b/w [0, 1]`, i.e. as cosine approaches 90 degrees, they approach 0, i.e. they are not at all similar, whereas the closer they are, the more they approach 0.

code: `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`

Cosine is directly proportional to similarity.

## Use Cases:
We can use, relationship between known relationships in a vector space to find out unknown relations, which can prove to be very useful. Simple arithmetic manipulation can help us evaluate and manipulate these relationships.
This can allow us to predict relationships.

## Reduction of Dimensions to Visualise Vector Space:
### PCA - Principle Component Analysis:
We implement dimension reduction while retaining as much information as possible, by reducing dimension to 2d, we are able to visualise data.

Eigenvector: Uncorrelated features
Eigenvalue: amount of information retained by each feature.

#### Algorith:
1. Normalize Data
2. Get Covariance Matrix
3. Perform Singular Value Decomposition

Obviously not all information would be retained and therefore, to calculate the amount of information retained, we use the formula: summation( EIGENVALUES[i][i] )/summation( EIGENVALUES[j][j] ), where i ranges from 0 to 1, i.e. number of dimensions required and j from 0 to d, the actual dimensions of the matrix.

`Projected data = dot( data , Eigenvecore[:required_dims] )`
# RELATER

`RELATER` is a Python package consisting of the 
implementation of the publication **Unsupervised 
Graph-based Entity Resolution for Complex Entities****
which is under review.

Entity Resolution (ER) is the process of linking 
records of the same entity across one or more 
databases in the absence of unique entity 
identifiers. `RELATER` is an unsupervised graph-based 
entity resolution framework that is focused on resolving 
the challenges associated with resolving complex entities. 
We propose a global method to propagate link decisions by 
propagating attribute values and constraints to 
capture changing attribute values and different 
relationships, a method for leveraging ambiguity in 
the ER process, an adaptive method of incorporating 
relationship structure, and a dynamic refinement step 
to improve record clusters by removing likely wrong 
links. `RELATER` can be employed to resolve records of 
both basic and complex entities.


## Usage

To run the `RELATER` framework on bibliographic data, for
example DBLP-ACM data set, you should run the following;
  
```bash
python -m er.bib_er dblp-acm1 $t_a $t_b $t_m $gamma $t_n
```
where $t_a, $t_b, $t_m, $gamma, and $t_n are the atomic 
node threshold, bootstrapping threshold, merging threshold,
the weight distribution in Equation (3), and threshold for
minimum number of nodes in a cluster to split by bridges,
respectively. 

## Settings

### Temporal Constraints

Based on the domain knowledge as specified in the paper, we set the following
temporal constraints. Since the constraints are data set 
specific, we show these constraints based on each data set.
 
##### IOS and KIL

* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Bm) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>) &ge; 55) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Bf) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mm) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
IsAfter(r<sub>i</sub>, r<sub>j</sub>) ^ AlmostSameBirthYears(r<sub>i</sub>, 
r<sub>j</sub>) &rarr; ValidMerge (r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Ds) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Dp) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mbp) ^ (30 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mgp) ^ (30 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bp) ^ (r<sub>j</sub>.&rho; = Bp) ^ (9 &le; 
MonthTimeGap(r<sub>i</sub>, r<sub>j</sub>)) ^ AlmostSameMarriageYears(r<sub>i</sub>, r<sub>j</sub>) &rarr; ValidMerge
(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bp) ^ (r<sub>j</sub>.&rho; = Mm) ^ AlmostSameMarriageYears(r<sub>i</sub>, r<sub>j</sub>) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bm) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
IsAfter(r<sub>i</sub>, r<sub>j</sub>) &rarr; ValidMerge(r<sub>i</sub>, 
r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bf) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(9 &le; MonthTimeGap(r<sub>i</sub>, r<sub>j</sub>) ) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mm) ^ (r<sub>j</sub>.&rho; = Mm) ^ 
AlmostSameBirthYears(r<sub>i</sub>, r<sub>j</sub>) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mm) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
IsAfter(r<sub>i</sub>, r<sub>j</sub>) ^ AlmostSameBirthYears(r<sub>i</sub>, 
r<sub>j</sub>) &rarr; ValidMerge (r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mm) ^ (r<sub>j</sub>.&rho; = Mbp) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mm) ^ (r<sub>j</sub>.&rho; = Mgp) ^ (15 &ge; 
YearTimeGap(r<sub>i</sub>, r<sub>j</sub>)) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)


### Link Constraints

Based on the domain knowledge as specified in the paper, we set the following
link constraints. Since the constraints are data set specific, 
we show these constraints based on each data set.
 
##### IOS and KIL

* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) ^ (|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Bp) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mm) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mbp) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Mgp) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Ds) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bb) ^ (r<sub>j</sub>.&rho; = Dp) ^ 
(|Links(r<sub>j</sub>,Bb)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Bp) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mm) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mbp) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Mgp) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Ds) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = Dp) ^ (r<sub>j</sub>.&rho; = Dd) ^ 
(|Links(r<sub>i</sub>,Dd)| = 0) &rarr; ValidMerge(r<sub>i</sub>, r<sub>j</sub>)

##### IPUMS
* (r<sub>i</sub>.&rho; = F) ^ (r<sub>j</sub>.&rho; = F) ^ 
(|Links(r<sub>i</sub>,F)| = 0) ^ (|Links(r<sub>j</sub>,F)| = 0) &rarr; 
ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = M) ^ (r<sub>j</sub>.&rho; = M) ^ 
(|Links(r<sub>i</sub>,M)| = 0) ^ (|Links(r<sub>j</sub>,M)| = 0) &rarr; 
ValidMerge(r<sub>i</sub>, r<sub>j</sub>)
* (r<sub>i</sub>.&rho; = C) ^ (r<sub>j</sub>.&rho; = C) ^ 
(|Links(r<sub>i</sub>,C)| = 0) ^ (|Links(r<sub>j</sub>,C)| = 0) &rarr; 
ValidMerge(r<sub>i</sub>, r<sub>j</sub>)


## Package structure
| Directory | Contains.. |
|---------------------|--------------------------------------------------------|
| common/ | Utility functions 
| data/        | Methods to retrieve and pre process data |
| er/  | ER algorithms proposed in `RELATER`           |
| febrl/ | Methods to calculate similarities from `febrl`           |


## Dependencies

The `RELATER` package requires the following python packages to be installed:
- [Python 2](http://www.python.org)
- [Networkx](https://networkx.org/)
- [Pandas](http://www.scipy.org)



## Contact

Contact the author of the package: [nishadi.kirielle@anu.edu.au](mailto:nishadi.kirielle@anu.edu.au)


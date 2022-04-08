import gdown
url = 'https://doc-04-1g-docs.googleusercontent.com/docs/securesc/sebhgnmd6slgvu9vv0podp9qksd3eg29/ca2u022eab4laorsil9fecv1dm8p632a/1648922475000/16237774210876738761/01943134975869900205/1euw3bygLzRQm_dYj1FoRduXvsRRUG2Gr?e=download&ax=ACxEAsbb7-7zSIODApmES-5d8-yaDAnEDm64sNy7DDVxhaWe3Ast26YTR3r-okztUnaBl1M'
output = 'ModelNet40_mvcnn_gvcnn.mat'
gdown.download(url,output,quiet=False)

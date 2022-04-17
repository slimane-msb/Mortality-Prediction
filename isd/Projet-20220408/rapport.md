# problem rencountre:

## plt.barplot  
* etant donne qu'il y a __pleusieur langue, et souvent avec moins de population__, notre graph (plt.bar) a tandance afficher tous a 0, et donc on a moins d'information a voir, et c'est pour cela qu'on reduit le graph auqulle il y a au moins un personne qui la parle, c'est le cas d'anglais "langue majoritaire" et espagnole et russe et ptun

## correlation entre certaines variables et la cible:
* pour eviter de mettre toutes les variable dans une seule figure, ce qui rend les __noms des variables illisible__, nous les avons separés en pleusieur figure, avec 15 variables pour chaque 

## l'ensemble de train/test/validation
* Premierement pour debugger les erreurs, et afin de ne pas attendre __30 min__ pour savoir qu'on fait une erreur de syntax par exemple, nous avons reduit l'ensemble de data a 400 lignes, ce qui permet une execution rapide, et ensuit la remettre a son etat original et dans ce cas pour ameliorer les performance. 


## balanced accuracy score a 50%:
* pas encore resolu


## temps d'execution tres lent :
* problem: a partir de +400 ligne, le temps d'execution devient lent, notament pour le model svc
* solution proposé: utilisation des threads pour partager l'excution 
* utiliser les GPU pour une excution Machine learning 
* reduire le nombre de ligne pour le training set 


## scater plot pour les deux premiere dimensions de la pca: 
* problem: 
    1. sans couleur 
    2. comment extraire des information 



## cross validation erreurs: 
1. The least populated class in y has only 6 members, which is less than n_splits=10. 
2. the least populated class in y has only 6 members, which is less than n_splits=10
* pas encore resolu: 






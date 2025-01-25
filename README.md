# Paper recommendation system
Link to our website: https://mingpassawat-datasciproj-streamlit2-yocvhr.streamlit.app/ <br>
The Presentation: https://youtu.be/Rk_KDCh7id0

## Guide ( Must Read )!
This is a part of Data Science project. This Github repository contained all the code and data using in this project. 
Here is a guide for this,

 - - We use `Streamlit2.py` to run the website on.
   		- This website use 3 csv files. Every csv contain basic info of the paper, but is use for different model.
   			- **outnow.csv**: Also contain 5-dimension vector.
   			- **outnow2.csv**: contain title,keyword for tf-idf.
   			- **outnow3.csv**: contain 100-dimension vector.

	
-  `/Fetching/main.py` :  Use for fetch extra 1000+ papers from Scopus search api.
	- `Fetchnow.csv` is a csv result dummy coding the fetched data, which later will be concat with the main data.
- `DATA_CSV/` contain all the provided data from CU into a sql-like schema csv.
	- `data.csv` contain is the main table
	- `affi_relation.csv, author_relation.csv` contain key
	- `authors.csv, affils.csv` contain information according to its id.
- `DataPrepNotebook/`
	- `tf.ipynb`: is used for prepare a data for tf-idf like combining keywords and title into one long string.
	- `vec.ipynb` : tokenize, and vectorize the keyword, and title.

# Ticket generation code

### Df_provider
*Df_provider* is the class that reads datasets that will be used to create tickets. These data are saved as csv in the folder *ticket_generation/data*
For each type of ticket, there is a class *df_provider* ( Ex: *complaint_df_provider* ) that inherit from the base class *df_provider*. In each of these inherited classes, the method *_preprocess_dataset* is implemented, which preprocess the columns of the csv file based on our needs ( Ex: change name of colunmn, change type of column...)
The purpose of this classes was not to change the original csv data, and to modify little things dinamically

### Employee_generator
*Employee_generator* is the class that generates an employee with all of his features. The features come from two sources: Faker, which creates all the random features (name, surname, mail...), and the df_providers, which give all the features specific to the category. Therefore, for example if the ticket request we want to generate is a request of salary raise, then the output of the *Employee_generator* will be something like {"name":"Gabriele", "surname":"Gioetto", ..., "prev_salary": $40000, "new_salary": $46000, ... }
In the *generate_<>_df* functions ( Ex. *generate_info_accommodation_df* ) the data are sampled from the output of the *df_providers* and processed ( Ex: add diff privacy, create new features, modify some features... ).
One of the main reason to do all this processing here and not in the *df_provider* classes is that in some cases the personal information of the employee are needed for the computations ( Ex: in *generate_refund_travel_df* the country of the employee is needed to generate the *origin* of the travel)

### Text_generator
*Text_generator* is the class that generates the text of the ticket from an employees dataframe ( the ones generated by *Employee_generators* ).
The main method is *generate_tickets*, which takes the *employees_df* and the templates of the tickets from the json file in *ticket_generation\data\templates*, iterates over all the employees in *employees_df* and for each employee create a ticket's text, which will be based on his features. The method also find heuristically the entities in the tickets' texts.
The *fine_tune* folder contains the methods to fine-tune the generation of the tickets' texts on the *Enron Mail dataset*

### Evaluate
*Evaluate_text* is used to evaluate the tickets' texts generated using different metrics
The *load_datasets* file is used to load the datasets which we use to compare the metrics.
The metrics used are:
- average TTR (Type Token Ratio) of unigrams
- average TTR (Type Token Ratio) of bigrams
- average ratios of nouns
- average ratios of verbs
- average word frequency ( Using a dump of wikipedia as reference )
- average word count
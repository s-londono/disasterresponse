from plotly.graph_objs import Bar, Box


class PlotBuilder:
    """
    Creates plots from a DataFrame containing categorized messages

    Attributes:
        df (DataFrame) Source of data to build plots
    """
    def __init__(self, df):
        self.df = df

    def build_genre_totals_bar(self):
        """
        Creates a bar plot depicting the number of messages per genre
        :return: Object definining the plot, in Plotly format
        """
        genre_counts = self.df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)

        return {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    def build_category_totals_bar(self):
        """
        Creates a bar plot of the total number of messages in each category
        :return: Object defining the plot, in Plotly format
        """
        df_category_totals = self.df.iloc[:, 4:].sum().sort_values(ascending=False)
        category_counts = df_category_totals.values
        category_names = [str(ix) for ix in df_category_totals.index]

        return {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                )
            ],
            'layout': {
                'title': 'Total Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45,
                    'type': 'category'
                }
            }
        }

    def build_message_length_box(self):
        """
        Creates a bar plot of the total number of messages in each category
        :return: Object defining the plot, in Plotly format
        """
        boxes = []

        for genre in self.df["genre"].unique():
            category_message_lengths = self.df[self.df['genre'] == genre]['message'].apply(len).values
            boxes.append(Box(x=category_message_lengths, name=genre))

        return {
            'data': boxes,
            'layout': {
                'title': 'Distribution of Message Lengths per Genre',
                'xaxis': {
                    'title': "Message Length"
                }
            }
        }

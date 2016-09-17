FIRST_LINE_INDEX = 0
SECOND_LINE_INDEX = 1
INITIAL_PAGE_RANK_VALUE = 1


def read_article_links(file_name):
    """this function will return a list of tuples, from the string representing a given text file"""
    with open(file_name, 'r', )as file:
        file_string = file.read()
        seperate_lines = file_string.split('\n')
        text_list = []
        for i in seperate_lines:
            if i != '':
                seperate_a_line = i.split('\t')
                couple = seperate_a_line[FIRST_LINE_INDEX], seperate_a_line[SECOND_LINE_INDEX]
                text_list.append(tuple(couple))
        return text_list


class Article:
    """this class will represent each article in a text file"""

    def __init__(self, name):
        self._name = name
        self._collection = []
        self._page_rank = INITIAL_PAGE_RANK_VALUE
        self._next_page_rank = INITIAL_PAGE_RANK_VALUE
        self._entry_level = 0
        self._in_neighbors = []

    def get_name(self):
        return self._name

    def add_neighbor(self, neighbor):
        "this function will add an article to self._collection"
        if neighbor not in self._collection:
            self._collection.append(neighbor)
            neighbor._entry_level += 1
            neighbor._in_neighbors.append(self)

    def get_neighbors(self):
        return self._collection

    def __repr__(self):
        neighbor_list = []
        if self.get_neighbors() == []:
            neighbor_list = []
        else:
            for article in self.get_neighbors():
                neighbor_list.append(article.get_name())
        article_repre = self._name, neighbor_list
        return str(article_repre)

    def __len__(self):
        return len(self._collection)

    def __contains__(self, article):
        if article in self._collection:
            return True
        else:
            return False


class WikiNetwork:
    """this class holds all the Article object of all the articles."""

    def __init__(self, link_list=[]):
        self._collection = {}
        self.update_network(link_list)

    def update_network(self, link_list):
        """gets an updated list of articles, and updates the current object's link list"""
        # runs on all the article tuples in link list
        for article_tuple in link_list:
            x = article_tuple[FIRST_LINE_INDEX].strip()
            y = article_tuple[SECOND_LINE_INDEX].strip()
            article_tuple = (x, y)
            # in case the first of a tuple not in the dictionary allready
            if article_tuple[FIRST_LINE_INDEX] not in self._collection:
                new_dict = {article_tuple[FIRST_LINE_INDEX]: Article(article_tuple[FIRST_LINE_INDEX])}
                self._collection.update(new_dict)
                # in case the second of a tuple is not in the dictionary allready
                if article_tuple[SECOND_LINE_INDEX] not in self._collection:
                    new_dict = {article_tuple[SECOND_LINE_INDEX]: Article(article_tuple[SECOND_LINE_INDEX])}
                    self._collection.update(new_dict)
                    self._collection[article_tuple[FIRST_LINE_INDEX]]. \
                        add_neighbor(self._collection[article_tuple[SECOND_LINE_INDEX]])
                # in case it is the the dictionary
                else:
                    self._collection[article_tuple[FIRST_LINE_INDEX]]. \
                        add_neighbor(self._collection[article_tuple[SECOND_LINE_INDEX]])
            # in case the first of a tuple is in the dictionary
            else:
                # in case the second of a tuple is not in the dictionary allready
                if article_tuple[SECOND_LINE_INDEX] not in self._collection:
                    new_dict = {article_tuple[SECOND_LINE_INDEX]: Article(article_tuple[SECOND_LINE_INDEX])}
                    self._collection.update(new_dict)
                    self._collection[article_tuple[FIRST_LINE_INDEX]]. \
                        add_neighbor(self._collection[article_tuple[SECOND_LINE_INDEX]])
                # in case it is the the dictionary
                else:
                    self._collection[article_tuple[FIRST_LINE_INDEX]]. \
                        add_neighbor(self._collection[article_tuple[SECOND_LINE_INDEX]])

    def get_articles(self):
        """returns a list of all the articles in the object"""
        list_of_articles = []
        for article in self._collection.values():
            list_of_articles.append(article)
        return list_of_articles

    def get_titles(self):
        """returns a list of all the article names"""
        name_list = [name.get_name() for name in self.get_articles()]
        return name_list

    def __contains__(self, article_name):
        if article_name in self.get_titles():
            return True
        else:
            return False

    def __len__(self):
        return len(self.get_titles())

    def __repr__(self):
        return str(self._collection)

    def __getitem__(self, article_name):
        return self._collection[article_name]

    def page_rank_calculater(self, article, d=0.9):
        if len(article.get_neighbors()) != 0:
            page_rank_list_for_sum = []
            for _neighbor in article._in_neighbors:
                current_rank = _neighbor._page_rank / len(_neighbor.get_neighbors())
                page_rank_list_for_sum.append(current_rank)
            article._next_page_rank = (d * sum(page_rank_list_for_sum)) + (1 - d)

    def page_rank(self, iters, d=0.9):
        """this function will return a list of articles,sorted by their page rank"""
        # page_rank_list = []
        for iter in range(iters):
            for article in self._collection.values():
                self.page_rank_calculater(article, d)
            for article in self._collection.values():
                article._page_rank = article._next_page_rank

        page_rank_list = [(article.get_name(),article._page_rank) for article in self._collection.values()]
        sorted_list = sorted(sorted(page_rank_list, key=lambda x: x[0]
                                    , reverse=False), key=lambda x: x[1], reverse=True)
        list_of_ranks = []
        for name in sorted_list:
            list_of_ranks.append(name[FIRST_LINE_INDEX])
        return list_of_ranks

    def jaccard_index(self, article_name):
        """this function will return a list of article names, sorted by their 'jaccard_index' with a given
        article name"""
        jaccard_index_list = []

        if article_name not in self.get_titles() or self._collection[article_name].get_neighbors() == []:
            return None
        else:
            for item in self.get_articles():
                the_neighbors_together = self._collection[article_name].get_neighbors()
                current_intersection = len(set(the_neighbors_together)
                                           .intersection(set(item.get_neighbors())))
                current_union = len(set(the_neighbors_together)
                                    .union(set(item.get_neighbors())))
                current_jaccard_index = current_intersection / current_union
                current_tuple = (item.get_name(), current_jaccard_index)
                jaccard_index_list.append(current_tuple)
            sorted_list = sorted(sorted(jaccard_index_list, key=lambda x: x[0]
                                        , reverse=False), key=lambda x: x[1], reverse=True)
            list_of_names_by_jaccard_index = []
            for name in sorted_list:
                list_of_names_by_jaccard_index.append(name[FIRST_LINE_INDEX])
            return list_of_names_by_jaccard_index
        
    def travel_path_iterator(self, article_name):
        """this function is a generator, that will go through all the articles in a
        given 'article_name' path"""
        current_article = self._collection.get(article_name)
        if current_article is None:
            return iter([])
        yield current_article.get_name()
        while current_article.get_neighbors():
            list_entry_level = []
            for neighbor in self._collection[current_article.get_name()].get_neighbors():
                current_tuple = (neighbor.get_name(), neighbor._entry_level)
                list_entry_level.append(current_tuple)
            sorted_list = sorted(sorted(list_entry_level, key=lambda x: x[0]
                                        , reverse=False), key=lambda x: x[1], reverse=True)
            if sorted_list != []:
                current_article = self._collection[sorted_list[0][0]]
            else:
                return
            yield current_article.get_name()

        #yield current_article
    def calculate_friends_by_depth(self, article_name, set_of_friends, depth):
        if depth == 0:
            set_of_friends.add(article_name)
            return set_of_friends
        set_of_friends.add(article_name)
        for article in self._collection[article_name].get_neighbors():
            set_of_friends.add(article.get_name())
            self.calculate_friends_by_depth(article.get_name(), set_of_friends, depth - 1)

    def friends_by_depth(self, article_name, depth):
        """this function will return a list with all the path neighbors of a given
        article, for a given depth size"""
        set_of_friends = set()
        if article_name not in self.get_titles():
            return None
        else:
            self.calculate_friends_by_depth(article_name, set_of_friends, depth)
            return list(set_of_friends)
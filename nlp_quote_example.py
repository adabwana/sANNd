from pynn import Base  # Import the pyNN Base class

#! Suggested by ChatGPT o3-mini-high

# Define a movie quote responder node as a dictionary.
# Its term function checks for certain keywords and returns a quote.
nMovieQuote = {
    "name": "movie quote responder",
    "term": lambda self, input: (
         "May the Force be with you." if "star wars" in input.lower() else
         "I'm gonna make him an offer he can't refuse." if "godfather" in input.lower() else
         "There is no spoon." if "matrix" in input.lower() else
         "Here's looking at you, kid."
    )
}

# Create an input/output node that simply passes along the input.
nIO = Base(
    name="i/o node",
    term=lambda self, input: input  # identity function
)

# Connect the movie quote responder node to the I/O node.
nIO.connect(Base, nMovieQuote)

# Main execution: prompt for input and print the resulting movie quote.
if __name__ == "__main__":
    user_input = input("Enter a movie reference (e.g., 'Star Wars', 'Godfather', 'Matrix'): ")
    output_quote = nIO(user_input)
    print("Movie Quote:", output_quote)

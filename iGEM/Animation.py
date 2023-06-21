from manim import *

## Currently Not Working, I haven't learn the library yet
class SIRModel(Scene):
    def construct(self):
        # Set up the SIR model parameters
        beta = 0.2  # transmission rate
        gamma = 0.1  # recovery rate

        # Set up initial population values
        S = [990.0]  # susceptible population
        I = [10.0]  # infectious population
        R = [0.0]  # recovered population

        # Set up the time grid
        t_max = 100  # maximum time
        dt = 0.1  # time step
        t = np.arange(0, t_max, dt)

        # Calculate the population values over time
        for i in range(1, len(t)):
            dS = -beta * S[i-1] * I[i-1]
            dI = beta * S[i-1] * I[i-1] - gamma * I[i-1]
            dR = gamma * I[i-1]
            S.append(S[i-1] + dS * dt)
            I.append(I[i-1] + dI * dt)
            R.append(R[i-1] + dR * dt)

        # Create the S, I, and R curves
        s_curve = self.get_curve(t, S)
        i_curve = self.get_curve(t, I)
        r_curve = self.get_curve(t, R)

        # Create the graph
        graph = self.get_graph(t, s_curve, i_curve, r_curve)

        # Add the graph to the scene
        self.add(graph)

    def get_curve(self, t, y):
        points = [np.array([x, y]) for x, y in zip(t, y)]
        curve = VGroup(
            *[Dot(point, color=BLUE) for point in points],
            smooth=False
        )
        return curve

    def get_graph(self, t, s_curve, i_curve, r_curve):
        graph = Axes(
            x_range=(0, t[-1]),
            y_range=(0, max(max(s_curve), max(i_curve), max(r_curve))),
            x_length=8,
            y_length=5,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [0, t[-1]]},
            y_axis_config={"numbers_to_include": [0, 1000, 2000]},
        )
        graph.to_edge(LEFT, buff=1)

        s_graph = graph.get_line_graph(s_curve, color=GREEN)
        i_graph = graph.get_line_graph(i_curve, color=RED)
        r_graph = graph.get_line_graph(r_curve, color=BLUE)

        graph.add(s_graph, i_graph, r_graph)
        return graph

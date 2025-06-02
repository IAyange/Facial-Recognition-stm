import 'package:flutter/material.dart';
import '../screens/home/home.dart';

class PersonalizationScreen extends StatefulWidget {
  final VoidCallback onComplete;
  const PersonalizationScreen({Key? key, required this.onComplete})
      : super(key: key);

  @override
  State<PersonalizationScreen> createState() => _PersonalizationScreenState();
}

class _PersonalizationScreenState extends State<PersonalizationScreen> {
  @override
  void initState() {
    super.initState();
    // Start countdown when screen loads
    _startPersonalization();
  }

  void _startPersonalization() {
    // Simulate personalization process for 5 seconds
    Future.delayed(const Duration(seconds: 5), () {
      // Navigate to home and dispose all previous routes
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (context) => const HomePage()),
        (Route<dynamic> route) => false,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primary,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(
              color: Colors.white,
              strokeWidth: 6,
            ),
            const SizedBox(height: 30),
            Text(
              "Analyzing your responses...",
              style: TextStyle(
                fontSize: 22,
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              "Creating your personalized dashboard",
              style: TextStyle(
                fontSize: 16,
                color: Colors.white.withOpacity(0.8),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              "Ensuring everything is captured perfectly",
              style: TextStyle(
                fontSize: 14,
                color: Colors.white.withOpacity(0.6),
              ),
            ),
            const SizedBox(height: 30),
            // Animated dots for extra flair
            _buildLoadingDots(),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingDots() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(3, (index) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeInOut,
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.5 + (index * 0.15)),
              shape: BoxShape.circle,
            ),
          ),
        );
      }),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:introduction_screen/introduction_screen.dart';
import 'package:lottie/lottie.dart';
import 'package:shared_preferences/shared_preferences.dart';

class OnboardingScreen extends StatefulWidget {
  final VoidCallback onComplete;

  const OnboardingScreen({Key? key, required this.onComplete})
      : super(key: key);

  @override
  _OnboardingScreenState createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final GlobalKey<IntroductionScreenState> _introKey = GlobalKey();

  Future<void> _completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboardingComplete', true);
    widget.onComplete(); // Call the onComplete callback
  }

  @override
  Widget build(BuildContext context) {
    const PageDecoration pageDecoration = PageDecoration(
      titleTextStyle: TextStyle(
        fontSize: 28.0,
        fontWeight: FontWeight.w700,
      ),
      bodyTextStyle: TextStyle(fontSize: 19.0),
      bodyPadding: EdgeInsets.fromLTRB(16.0, 0.0, 16.0, 16.0),
      pageColor: Colors.white,
      imagePadding: EdgeInsets.zero,
    );

    return IntroductionScreen(
      key: _introKey,
      globalBackgroundColor: Colors.white,
      pages: [
        PageViewModel(
          title: "Welcome to RaxCare",
          body:
              "First Of all congratulations, it is not easy for a person to accept that they are in such a situation that you are in. This is Your trusted companion on the journey to addiction recovery and wellness.",
          image: Lottie.asset(
            'assets/animations/welcome.json',
            height: 300,
          ),
          decoration: pageDecoration,
        ),
        PageViewModel(
          title: "Track Your Progress",
          body:
              "Monitor your recovery journey with personalized tracking tools and insights.",
          image: Lottie.asset(
            'assets/animations/progress.json',
            height: 300,
          ),
          decoration: pageDecoration,
        ),
        PageViewModel(
          title: "AI Analytics",
          body:
              "Leverage advanced AI analytics to gain deeper insights into your progress and make data-driven decisions.",
          image: Lottie.asset(
            'assets/animations/ai_analytics.json',
            height: 300,
          ),
          footer: Padding(
            padding: const EdgeInsets.only(top: 24.0),
            child: ElevatedButton(
              onPressed: _completeOnboarding,
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.white,
                minimumSize: const Size(200, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8.0),
                ),
              ),
              child: const Text(
                'GET STARTED',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
          ),
          decoration: pageDecoration,
        ),
      ],
      onDone: _completeOnboarding,
      showSkipButton: true,
      skipOrBackFlex: 0,
      nextFlex: 0,
      skip: const Text('Skip'),
      next: const Icon(Icons.arrow_forward),
      done: const Text('Done', style: TextStyle(fontWeight: FontWeight.w600)),
      curve: Curves.fastLinearToSlowEaseIn,
      controlsMargin: const EdgeInsets.all(16),
      controlsPadding: const EdgeInsets.fromLTRB(8.0, 4.0, 8.0, 4.0),
      dotsDecorator: const DotsDecorator(
        size: Size(10.0, 10.0),
        color: Color(0xFFBDBDBD),
        activeSize: Size(22.0, 10.0),
        activeShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(25.0)),
        ),
      ),
    );
  }
}

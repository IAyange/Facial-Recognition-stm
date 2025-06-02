import 'package:RaxCare/authentication/login.dart';
import 'package:RaxCare/authentication/register.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/material.dart';

import 'onboarding_screen.dart';
import 'personalization_screen.dart';
import '../screens/home/home.dart';

class Authenticate extends StatefulWidget {
  const Authenticate({Key? key}) : super(key: key);

  @override
  _AuthenticateState createState() => _AuthenticateState();
}

class _AuthenticateState extends State<Authenticate> {
  bool showSignIn = true;
  bool onboardingComplete = false;
  bool isLoading = true;
  bool isNewUser = false;

  @override
  void initState() {
    super.initState();
    _checkOnboarding();
    _checkIfNewUser();
  }

  Future<void> _checkOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    final completed = prefs.getBool('onboardingComplete') ?? false;
    setState(() {
      onboardingComplete = completed;
      isLoading = false;
    });
  }

  Future<void> _checkIfNewUser() async {
    final prefs = await SharedPreferences.getInstance();
    final isNew = prefs.getBool('isNewUser') ?? true;
    setState(() {
      isNewUser = isNew;
    });
  }

  Future<void> _completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboardingComplete', true);
    setState(() {
      onboardingComplete = true;
    });
  }

  void toggleView() {
    setState(() {
      showSignIn = !showSignIn;
    });
  }

  Future<void> _handleNewUserRegistration() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isNewUser', false);
    setState(() {
      isNewUser = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    final user = FirebaseAuth.instance.currentUser;

    if (!onboardingComplete) {
      return OnboardingScreen(onComplete: _completeOnboarding);
    }

    if (user == null) {
      return AnimatedSwitcher(
        duration: const Duration(milliseconds: 300),
        transitionBuilder: (Widget child, Animation<double> animation) {
          return FadeTransition(opacity: animation, child: child);
        },
        child: showSignIn
            ? Login(toggleView: toggleView, key: const ValueKey('login'))
            : Register(
                toggleView: toggleView,
                key: const ValueKey('register'),
                onRegistrationComplete: () {
                  _handleNewUserRegistration();
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(
                      builder: (context) => PersonalizationScreen(
                        onComplete: _handleNewUserRegistration,
                      ),
                    ),
                  );
                },
              ),
      );
    }

    // For new users after registration
    if (isNewUser) {
      return PersonalizationScreen(
        onComplete: _handleNewUserRegistration,
      );
    }

    // For returning users
    return const HomePage();
  }
}

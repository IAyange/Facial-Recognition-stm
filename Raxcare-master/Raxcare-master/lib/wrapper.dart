// wrapper.dart
import 'package:RaxCare/authentication/authenticate.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:RaxCare/app.dart';
import 'package:RaxCare/models/the_user.dart';

class Wrapper extends StatelessWidget {
  const Wrapper({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final user = Provider.of<TheUser?>(context);

    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: user == null ? const Authenticate() : const App(),
    );
  }
}
